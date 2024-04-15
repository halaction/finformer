import json
import yaml


fmp_config = {
    'profile': {
        'dir': None,
        'endpoint': 'profile',
        'path_params': {
            'symbol': None,
        },
        'query_params': {
            'apikey': None,
        },
    },
    'news': {
        'dir': None,
        'endpoint': 'stock_news',
        'path_params': None,
        'query_params': {
            'page': None,
            'tickers': None,
            'limit': None,
            'apikey': None,
        },
    },
    'prices': {
        'dir': None,
        'endpoint': 'historical-price-full',
        'path_params': {
            'symbol': None,
        },
        'query_params': {
            'from': None,
            'to': None,
            'serietype': None,
            'apikey': None,
        },
    },
    'metrics': {
        'dir': None,
        'endpoint': 'key-metrics',
        'path_params': {
            'symbol': None,
        },
        'query_params': {
            'period': None,
            'limit': None,
            'apikey': None,
        },
    },
}


if __name__ == '__main__':

    with open('fmp-config.json', 'w', encoding='utf-8') as file:
        json.dump(fmp_config, file, indent=2)

    with open('finformer/data/fmp-config.yaml', 'w', encoding='utf-8') as file:
        yaml.safe_dump(fmp_config, file)

    with open('finformer/data/fmp-config.yaml', 'r', encoding='utf-8') as file:
        cfg = yaml.safe_load(file)

    print(cfg)
