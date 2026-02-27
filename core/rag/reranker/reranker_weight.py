from typing import List, Optional, Dict, Set, Tuple
from collections import Counter
import re
import math

import jieba.analyse
import numpy as np

from core.rag.reranker.reranker_base import BaseReranker
from core.rag.entities.document import Document
from core.rag.vectorizer.vectorize_processor import VectorizeProcessor
from core.rag.data.jieba_stopwords import STOPWORDS

class WeightReranker(BaseReranker):
    def __init__(self, weight: float, model_instance_provider: str, model_instance_config: Dict, **kwargs):
        '''
            weight 为语义权重
            此处使用的是Embedding模型
        '''
        self.semantics_weight = weight
        self.keyword_weight = 1 - weight
        self.vectorizer = VectorizeProcessor.get_vectorizer(model_instance_provider=model_instance_provider, model_instance_config=model_instance_config)
        
    async def rerank(
        self, 
        query: str, 
        documents: List[Document], 
        top_k: int, 
        score_threshold: Optional[float] = None
    ) -> Tuple[List[Document], Dict]:
        # documents 去重
        unique_documents = []
        content_id_set = set()
        for document in documents:
            if document.metadata is not None and 'content_id' in document.metadata.keys() and document.metadata['content_id'] not in content_id_set:
                content_id_set.add(document.metadata['content_id'])
                unique_documents.append(document)
        documents = unique_documents

        # 计算语义得分
        semantics_scores = [0.] * len(documents)
        usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        if self.semantics_weight > 0:
            semantics_scores, usage = await self._calculate_semantics_scores(query, documents)

        # 计算关键词得分
        keyword_scores = [0.] * len(documents)
        if self.keyword_weight > 0:
            keyword_scores = self._calculate_keyword_scores(query, documents)

        # 根据语义权重，计算总得分，并进行阈值筛选
        rerank_documents = []
        for document, semantics_score, keyword_score in zip(documents, semantics_scores, keyword_scores):
            score = self.semantics_weight * semantics_score + self.keyword_weight * keyword_score
            
            if score_threshold and score < score_threshold:
                continue

            if document.metadata is not None:
                document.metadata['score'] = score
                rerank_documents.append(document)

        # 降序排序
        rerank_documents.sort(key=lambda x: x.metadata["score"] if x.metadata else 0, reverse=True)

        # 返回top_k
        return rerank_documents[:top_k], usage

    async def _calculate_semantics_scores(self, query: str, documents: List[Document]) -> Tuple[List[float], Dict]:
        '''
            向量余弦相似度
        '''
        # 计算query的向量
        query_vectorize_result = await VectorizeProcessor.vectorize(self.vectorizer, [query])
        query_vector = query_vectorize_result[0]["vector"]

        if documents and any(doc.vector is None for doc in documents):
            contents = [document.page_content for document in documents]
            contents_vector = await VectorizeProcessor.vectorize(self.vectorizer, contents)

        # 计算与各document的余弦相似度
        scores: List[float] = []
        for idx, document in enumerate(documents):
            vec1 = np.array(query_vector)
            vec2 = np.array(document.vector if document.vector else contents_vector[idx]["vector"])

            # 两向量的点积 除以 两向量模长的乘积

            # calculate dot product
            dot_product = np.dot(vec1, vec2)

            # calculate norm
            norm_vec1 = np.linalg.norm(vec1)
            norm_vec2 = np.linalg.norm(vec2)

            # calculate cosine similarity
            cosine_sim = dot_product / (norm_vec1 * norm_vec2)
            scores.append(cosine_sim)
        
        return scores, query_vectorize_result[0]["usage"]

    def _calculate_keyword_scores(self, query: str, documents: List[Document]) -> List[float]:
        '''
            BM25
        '''
        # 提取query的关键词
        query_keywords: Set[str] = self._extract_keywords(query)

        # 提取documents的关键词
        documents_keywords_list: List[Set[str]] = []        # 每个文档的关键词
        documents_keywords: Set[str] = set()                # 全部文档的关键词
        for document in documents:
            document_keywords = self._extract_keywords(document.page_content)
            if document.metadata is not None:
                # document.metadata['keywords'] = document_keywords
                documents_keywords_list.append(document_keywords)
                documents_keywords.update(document_keywords)
            else:
                documents_keywords_list.append([])
        
        # 计算query中关键词的词频 TF
        query_keywords_tf = Counter(query_keywords)

        # 计算documents中关键词的逆文档频率 IDF
        documents_keywords_idf: Dict[str, float] = {}       # 全部文档的关键词的IDF
        documents_len = len(documents)
        for keyword in documents_keywords:
            # 计算在documents中有多少篇document包含了关键词keyword
            freq = sum(1 for document_keywords in documents_keywords_list if keyword in document_keywords)
            documents_keywords_idf[keyword] = math.log((1 + documents_len) / (1 + freq)) + 1

        # 计算query中关键词的 TF-IDF
        query_keywords_tfidf: Dict[str, float] = {}
        for query_keyword, tf in query_keywords_tf.items():
            idf = documents_keywords_idf.get(query_keyword, 0)
            query_keywords_tfidf[query_keyword] = tf * idf

        # 分别计算每个document中关键词的 TF-IDF
        documents_keywords_tfidf: List[Dict[str, float]] = []   # 每个文档的关键词的IDF
        for document_keywords in documents_keywords_list:
            document_keywords_tf = Counter(document_keywords)
            document_keywords_tfidf = {}
            for document_keyword, tf in document_keywords_tf.items():
                idf = documents_keywords_idf.get(document_keyword, 0)
                document_keywords_tfidf[document_keyword] = tf * idf
            documents_keywords_tfidf.append(document_keywords_tfidf)

        # 计算query与每个document的相似度
        def cosine_similarity(vec1, vec2) -> float:
            intersection = set(vec1.keys()) & set(vec2.keys())
            numerator = sum(vec1[x] * vec2[x] for x in intersection)

            sum1 = sum(vec1[x] ** 2 for x in vec1)
            sum2 = sum(vec2[x] ** 2 for x in vec2)
            denominator = math.sqrt(sum1) * math.sqrt(sum2)

            return float(numerator) / denominator if denominator else 0.0

        similarities: List[float] = []
        for document_keywords_tfidf in documents_keywords_tfidf:
            similarity = cosine_similarity(query_keywords_tfidf, document_keywords_tfidf)
            similarities.append(similarity)
        
        return similarities

    def _extract_keywords(self, text: str) -> Set[str]:
        keywords = jieba.analyse.extract_tags(sentence=text, topK=None, withWeight=False)
        keywords = set(keywords)

        results = set()
        for keyword in keywords:
            results.add(keyword)
            # 拆分英文、数字等子串
            sub_keywords = re.findall(r'\w+', keyword)
            if len(sub_keywords) > 1:
                results.update({sub_keyword for sub_keyword in sub_keywords if sub_keyword not in STOPWORDS})

        return results