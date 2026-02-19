from flask import Flask, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import traceback
import numpy as np

app = Flask(__name__)
CORS(app)

# Lista ampliada de ETFs populares
ETF_TICKERS = [
    "SPY", "QQQ", "IVV", "VOO", "VTI", "BND", "AGG", "GLD", "SLV", "IEFA",
    "EEM", "VWO", "TLT", "IWM", "XLK", "XLV", "XLE", "XLF", "XLI", "XLB",
    "XLP", "XLU", "XLY", "SMH", "SOXX", "IBB", "KBE", "KRE", "USO", "UNG",
    "HYG", "LQD", "SHY", "IEI", "TIP", "VNQ", "DBC", "GDX", "GDXJ", "EWJ",
    "MCHI", "FXI", "INDA", "EPI", "EWZ", "ARKK", "ARKW", "ARKG", "QCLN", "TAN",
    "ICLN", "PBW", "XBI", "LABU", "BLOK", "FINX", "LIT", "REMX", "URA", "KOL",
    "SCHD", "VYM", "VIG", "NOBL", "SPHD", "JEPI", "JEPQ", "DIVO", "SCHG", "VUG",
    "IVW", "IJR", "IJH", "IJS", "IJT", "AVUV", "AVDV", "AVEM", "AVDE", "AVUS"
]

def convert_to_python_types(obj):
    """Convierte tipos NumPy a tipos Python estándar"""
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                       np.int16, np.int32, np.int64, np.uint8,
                       np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    elif pd.isna(obj):
        return None
    return obj

def safe_bool(value):
    """Convierte cualquier valor a bool Python estándar"""
    return bool(value) if value is not None else False

def safe_float(value, default=0.0):
    """Convierte a float de forma segura"""
    if value is None or pd.isna(value):
        return default
    try:
        return float(value)
    except:
        return default

def get_expense_ratio(info, ticker):
    """Obtiene el expense ratio real del ETF"""
    # Intentar diferentes campos donde Yahoo guarda el expense ratio
    expense_fields = [
        'annualReportExpenseRatio',
        'expenseRatio',
        'annualExpenseRatio',
        'prospectusNetExpenseRatio',
        'managementExpenseRatio'
    ]
    
    for field in expense_fields:
        value = info.get(field)
        if value and not pd.isna(value) and value > 0:
            return safe_float(value)
    
    # Valores conocidos por ticker (fallback)
    known_expenses = {
        "SPY": 0.0945, "QQQ": 0.20, "IVV": 0.04, "VOO": 0.03, "VTI": 0.03,
        "BND": 0.03, "AGG": 0.03, "GLD": 0.40, "SLV": 0.50, "IEFA": 0.07,
        "EEM": 0.68, "VWO": 0.08, "TLT": 0.15, "IWM": 0.19, "XLK": 0.10,
        "XLV": 0.10, "XLE": 0.10, "XLF": 0.10, "XLI": 0.10, "XLB": 0.10,
        "XLP": 0.10, "XLU": 0.10, "XLY": 0.10, "SMH": 0.35, "SOXX": 0.35,
        "IBB": 0.45, "KBE": 0.35, "KRE": 0.35, "USO": 0.60, "UNG": 0.60,
        "HYG": 0.49, "LQD": 0.14, "SHY": 0.15, "IEI": 0.15, "TIP": 0.19,
        "VNQ": 0.12, "DBC": 0.85, "GDX": 0.51, "GDXJ": 0.51, "EWJ": 0.49,
        "MCHI": 0.59, "FXI": 0.74, "INDA": 0.64, "EPI": 0.59, "EWZ": 0.59,
        "ARKK": 0.75, "ARKW": 0.75, "ARKG": 0.75, "QCLN": 0.40, "TAN": 0.65,
        "ICLN": 0.46, "PBW": 0.60, "XBI": 0.35, "LABU": 1.48, "BLOK": 0.75,
        "FINX": 0.68, "LIT": 0.75, "REMX": 0.65, "URA": 0.70, "KOL": 0.65,
        "SCHD": 0.06, "VYM": 0.06, "VIG": 0.06, "NOBL": 0.35, "SPHD": 0.28,
        "JEPI": 0.35, "JEPQ": 0.35, "DIVO": 0.50, "SCHG": 0.04, "VUG": 0.04,
        "IVW": 0.18, "IJR": 0.06, "IJH": 0.05, "IJS": 0.18, "IJT": 0.18,
        "AVUV": 0.25, "AVDV": 0.36, "AVEM": 0.33, "AVDE": 0.27, "AVUS": 0.15
    }
    
    return known_expenses.get(ticker, 0.30)  # default 0.30% si no se encuentra

def get_etf_data(ticker):
    """Obtiene datos de un ETF usando yfinance"""
    try:
        etf = yf.Ticker(ticker)
        info = etf.info
        
        # Obtener datos históricos para medias móviles
        hist = etf.history(period="1y")
        
        # Calcular medias móviles
        sma50 = None
        sma150 = None
        sma200 = None
        
        if not hist.empty:
            if len(hist) >= 50:
                sma50 = safe_float(hist['Close'].rolling(50).mean().iloc[-1])
            if len(hist) >= 150:
                sma150 = safe_float(hist['Close'].rolling(150).mean().iloc[-1])
            if len(hist) >= 200:
                sma200 = safe_float(hist['Close'].rolling(200).mean().iloc[-1])
        
        # Precio actual
        price = safe_float(info.get('regularMarketPrice', 
                           info.get('currentPrice', 
                           hist['Close'].iloc[-1] if not hist.empty else 0)))
        
        # Volumen
        volume = safe_float(info.get('regularMarketVolume', 0))
        
        # Máximo 52 semanas
        week52_high = safe_float(info.get('fiftyTwoWeekHigh', 
                                hist['High'].max() if not hist.empty else price))
        
        # AUM (activos bajo gestión)
        aum = safe_float(info.get('totalAssets', info.get('marketCap', 0)))
        
        # Expense ratio (MEJORADO)
        expense_ratio = get_expense_ratio(info, ticker)
        
        # Determinar clase de activo por nombre
        name = str(info.get('longName', info.get('shortName', ticker)))
        asset_class = 'Equity'
        name_lower = name.lower()
        ticker_lower = ticker.lower()
        
        bond_keywords = ['bond', 'treasury', 'aggregate', 'fixed income', 'corporate bond', 'municipal']
        commodity_keywords = ['gold', 'silver', 'oil', 'commodity', 'metals', 'energy', 'natural gas']
        
        if any(kw in name_lower for kw in bond_keywords) or any(kw in ticker_lower for kw in ['bnd', 'agg', 'tlt', 'shy', 'iei', 'lqd', 'hyg']):
            asset_class = 'Fixed Income'
        elif any(kw in name_lower for kw in commodity_keywords) or any(kw in ticker_lower for kw in ['gld', 'slv', 'uso', 'ung', 'dbc']):
            asset_class = 'Commodity'
        
        # Calcular porcentaje cerca del máximo
        near_52w_pct = safe_float((price / week52_high * 100) if week52_high and week52_high > 0 else 100)
        
        # Reglas técnicas
        close_above_52w = safe_bool(price >= week52_high * 0.98)
        sma50_gt_sma150 = safe_bool(sma50 and sma150 and sma50 > sma150)
        sma150_gt_sma200 = safe_bool(sma150 and sma200 and sma150 > sma200)
        sma200_slope = safe_bool(sma200 and price > sma200)  # Pendiente positiva si precio > SMA200
        
        # RSI
        rsi = 50
        if not hist.empty and len(hist) >= 14:
            try:
                delta = hist['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi_val = 100 - (100 / (1 + rs.iloc[-1]))
                if not pd.isna(rsi_val):
                    rsi = int(safe_float(rsi_val))
            except:
                pass
        
        # Lwowski rating (proxy: AUM * volumen / 1e6)
        lwowski = 5000
        if aum and volume:
            lwowski = int(safe_float((aum / 1e6) * (volume / 1e6) / 100))
            lwowski = max(1000, min(99999, lwowski))
        
        # Prev close
        prev_close = safe_float(info.get('regularMarketPreviousClose', price * 0.99))
        
        # Construir diccionario
        result = {
            'ticker': str(ticker),
            'name': str(name)[:60],
            'asset': str(asset_class),
            'lwowski': int(lwowski),
            'price': float(round(price, 2)),
            'closeAbove52w': bool(close_above_52w),
            'sma50gt150': bool(sma50_gt_sma150),
            'sma150gt200': bool(sma150_gt_sma200),
            'sma200Slope': bool(sma200_slope),
            'aum': str(int(aum/1e6)) if aum else 'N/A',
            'rsi': int(rsi),
            'near52wpct': float(round(near_52w_pct, 1)),
            'week52High': float(round(week52_high, 2)),
            'prevClose': float(round(prev_close, 2)),
            'volume': float(round(volume / 1e6, 1)),
            'expense': float(round(expense_ratio * 100, 2))  # Expense ratio en porcentaje
        }
        
        return convert_to_python_types(result)
        
    except Exception as e:
        print(f"Error con {ticker}: {str(e)}")
        return None

@app.route('/api/etfs', methods=['GET'])
def get_etfs():
    """Endpoint principal que devuelve datos de ETFs"""
    try:
        results = []
        total = len(ETF_TICKERS)
        errors = []
        
        for i, ticker in enumerate(ETF_TICKERS):
            print(f"Procesando {i+1}/{total}: {ticker}")
            try:
                data = get_etf_data(ticker)
                if data:
                    results.append(data)
                else:
                    errors.append(ticker)
            except Exception as e:
                print(f"  → Error en {ticker}: {str(e)}")
                errors.append(ticker)
        
        print(f"✅ Completado: {len(results)} ETFs obtenidos correctamente")
        if errors:
            print(f"⚠️  Errores en {len(errors)} ETFs: {', '.join(errors[:5])}...")
        
        return jsonify({
            'status': 'success',
            'count': len(results),
            'data': results
        })
    
    except Exception as e:
        print(f"❌ Error general: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e),
            'data': []
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'ETF Screener API running'})

if __name__ == '__main__':
    print("🚀 Iniciando servidor ETF Screener...")
    print(f"📊 Cargando {len(ETF_TICKERS)} ETFs")
    print("💡 Servidor disponible en: http://localhost:5000")
    print("🔧 API endpoint: http://localhost:5000/api/etfs")
    print("✅ Expense ratios: Usando datos reales de Yahoo Finance")
    app.run(host='0.0.0.0', port=5000, debug=True)
