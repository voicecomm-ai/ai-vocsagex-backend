import json
import argparse
import traceback
import asyncio
from concurrent.futures import ThreadPoolExecutor

import uvicorn

from logger import get_logger
import api.api
from config.config import init_config, get_config

logger = get_logger('main')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--config_path',
        type=str,
        default='./config/config.json',
        help='Path to the server config.'
    )

    args = parser.parse_args()
    logger.info(f'Input arguments: {args}')

    try:
        init_config(args.config_path)
        logger.info(f'Server config:\n{json.dumps(get_config(), indent=4)}')
    except Exception as e:
        logger.fatal(f'Fail to load config file for:\n{traceback.format_exc()}')
        exit(-1)

    asyncio.run(start_server(**get_config()['server']))


async def start_server(**kwargs):
    host = kwargs['host']
    port = kwargs['port']
    default_executor_threads = kwargs['default_executor_threads']

    executor = ThreadPoolExecutor(max_workers=default_executor_threads)
    loop = asyncio.get_running_loop()
    loop.set_default_executor(executor)

    logger.info(f'Starting server on {host}:{port} with thread pool size {default_executor_threads}')
    config = uvicorn.Config(api.api.app, host=host, port=port)
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == '__main__':
    main()
