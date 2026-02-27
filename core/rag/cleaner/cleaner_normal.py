import re

from core.rag.cleaner.cleaner_base import BaseCleaner

class NormalCleaner(BaseCleaner):
    def __init__(self, **kwargs):
        if 'filter_blank' not in kwargs:
            raise ValueError('NormalCleaner: missing required parameter: "filter_blank".')

        if 'remove_url' not in kwargs:
            raise ValueError('NormalCleaner: missing required parameter: "remove_url".')

        self.filter_blank = kwargs['filter_blank']
        self.remove_url = kwargs['remove_url']

    def clean(self, text: str) -> str:
        # default clean
        # remove invalid symbol
        text = re.sub(r"<\|", "<", text)
        text = re.sub(r"\|>", ">", text)
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\xEF\xBF\xBE]", "", text)
        # Unicode  U+FFFE
        text = re.sub("\ufffe", "", text)

        if self.filter_blank:
            # 3+个\n都转为2个\n
            pattern = r"\n{3,}"
            text = re.sub(pattern, "\n\n", text)
            # 2+个空白符都转为1个空格
            pattern = r"[\t\f\r\x20\u00a0\u1680\u180e\u2000-\u200a\u202f\u205f\u3000]{2,}"
            text = re.sub(pattern, " ", text)
        
        if self.remove_url:
            # Remove email
            pattern = r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)"
            text = re.sub(pattern, "", text)

            # Remove URL but keep Markdown image URLs
            # First, temporarily replace Markdown image URLs with a placeholder
            markdown_image_pattern = r"!\[.*?\]\((https?://[^\s)]+)\)"
            placeholders: list[str] = []

            def replace_with_placeholder(match, placeholders=placeholders):
                url = match.group(1)
                placeholder = f"__MARKDOWN_IMAGE_URL_{len(placeholders)}__"
                placeholders.append(url)
                return f"![image]({placeholder})"

            text = re.sub(markdown_image_pattern, replace_with_placeholder, text)

            # 单独移除超链接，避免后续删除时，保留了超链接格式
            markdown_hyperlink_pattern = r'\[(.*?)\]\((https?://[^\s)]+)\)'
            text = re.sub(markdown_hyperlink_pattern, r'\1',text)

            # Now remove all remaining URLs
            url_pattern = r"https?://[^\s)]+"
            text = re.sub(url_pattern, "", text)

            # Finally, restore the Markdown image URLs
            for i, url in enumerate(placeholders):
                text = text.replace(f"__MARKDOWN_IMAGE_URL_{i}__", url)

        return text