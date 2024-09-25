**FX VOLATILITY SURFACE CONSTRUCTION WITH SABR MODEL**

**TARGET**

Construct EURUSD FX volatility surface.
- Convert quoted data (delta/maturity) in call/put volatilities (strike/tenor) and capture the volatility surface dynamics

**DATA**

Market volatilities quoted by data provider:
- in terms of ATM risk-reversal and market strangle strategies 
- using spot/forward delta and DNS ATM conventions depending on maturities.

**TECHNIQUE**

SABR model calibrated to fit market data. 
- Interpolation: calibrated SABR for interpolation along strikes & backbone flat-forward volatility approach for temporal interpolation.

**RESULTS**

High calibration precision.
