import requests

def get_solana_price():
    url = 'https://api.coingecko.com/api/v3/simple/price'
    params = {
        'ids': 'solana',
        'vs_currencies': 'usd'
    }
    response = requests.get(url, params=params)
    data = response.json()
    return data['solana']['usd']

# Exemplo de uso:
preco = get_solana_price()
print(f'Current Solana price: ${preco:.2f}')
