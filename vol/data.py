from typing import Union, Sequence
import pandas as pd
import numpy as np
import numpy.typing as npt
import xarray as xr
from scipy import interpolate as intp
from datetime import datetime 
from . import surface

Date =  Union[str, datetime.datetime]
DateList = Union[Date, sequence[Date]]

              
def load_moex_surface(date: Date,
                      base_asset: str,
                      expiration: DateList,
                      data_dir: str = 'data/moex'):
    """Load a volatility surface from the MOEX option board data saved on the 
    disk.
    
    Parameters
    ----------
    date : string or datetime
        The calendar date at which the prices are quoted. If a string, it must 
        be in the yyyy-mm-dd format.
    base : string
        The base futures contract, e.g. 'Si-9.22', 'RTS-12.22', etc.
    expiration : string or datetime or list of strings or list of datetime
        Expiration date(s) of options. If a string or a list of strings, it must 
        be in the yyyy-mm-dd format.    
    data_dir : string 
        folder with data (see the notes below)
        
    Returns
    -------
    moex_surface : xarray
        volatility surface as xarray with coordinates 'tau' (time to expiration) 
        and 'x' (strike)
        
    Notes
    -----
    * Data structure
    
    This function loads data from the MOEX option board, which can be (manually)
    downloaded and saved as csv files in data_dir directory. The directory 
    structure must contain files named as follows:
    
        {data_dir}/{date}/moex-{base}-{expiration}-marg-optiondesk.csv
    
    where date_dir, date, base are the same as the function's parameters (date
    is in yyyy-mm-dd format), and expiration in the expiration date in ddmmyy 
    format (this is done for convenience, since this is how MOEX names the files 
    by default). These files can be downloaded from MOEX's website
    
        https://www.moex.com/ru/derivatives/optionsdesk.aspx
        
    Besides that, for each date in the data_dir, there should be the file
    
         {data_dir}/{date}/futures.csv
    
    which should be created manually and provides the corresponding futures price. 
    This file must contain two columns:
    
        contract, price
        
    where contract is the contract's name, which is the same as 'base' variable, 
    price is the futures price at date 'date'. 
    
    * How implied volatility is computed
    
    Currently we simply use the implied volatility provided by the exchange, in 
    the 'IV' column of their csv files. Be careful: it is not guaranteed to be 
    arbitrage-free.
    
    * Return value
    
    The volatility surface is stored and returned as a 2-D xarray instance. There
    are two dimensions with labels 'tau' and 'k', where tau is the time to 
    expiration in fractions of a year and k in the strike. The xarray also has the 
    following attributes:
    
    date : float
        calendar date (same as date parameter of the function)
    base_price : float
        futures price         
    """
      
    t = datetime.strptime(date, "%Y-%m-%d") if type(date) is str else date
    base_price = pd.read_csv("{dir}/{date}/futures.csv".format(dir=data_dir, date=t.strftime("%Y-%m-%d")), 
        index_col="contract")["price"][base]
    
    curves = []
    for e in expiration if type(expiration) is list else [expiration]:        
        T = datetime.strptime(e, "%Y-%m-%d") if type(e) is str else e
        filename = "{dir}/{date}/moex-{base}-{expir}-marg-optionsdesk.csv".format(dir=data_dir, 
            date=t.strftime("%Y-%m-%d"), base=base, expir=T.strftime("%d%m%y"))
        data = pd.read_csv(filename, sep=",", index_col="СТРАЙК", encoding="CP1251")
        data.rename(columns = {
                #"CALL: Объем торгов, руб" : "call_volume_rub",
                #"CALL: Объем торгов, контр" : "call_volume_contracts" ,
                #"CALL: Открыт.позиций" : "call_open_interest",
                "CALL: Последняя сделка, Значение" : "call_last_price", 
                #"CALL: Последняя сделка, Дата" : "call_last_date",
                #"CALL: Последняя сделка, Изменение": "call_change",
                #"CALL: ПОКУПКА" : "call_bid",
                #"CALL: ПРОДАЖА" : "call_ask", 
                "CALL: Расчетная цена" : "call_settlement_price",
                #"CALL: Теоретическая цена" : "call_theoretical_price",
                "IV" : "iv",                
                #"PUT: Объем торгов, руб" : "put_volume_rub",
                #"PUT: Объем торгов, контр" : "put_volume_contracts" ,
                #"PUT: Открыт. позиций" : "put_open_interest",
                "PUT: Последняя сделка, Значение" : "put_last_price", 
                #"PUT: Последняя сделка, Дата" : "put_last_date",
                #"PUT: Последняя сделка, Изменение": "put_change",
                #"PUT: ПОКУПКА" : "put_bid",
                #"PUT: ПРОДАЖА" : "put_ask", 
                "PUT: Расчетная цена" : "put_settlement_price",
                #"PUT: Теоретическая цена" : "put_theoretical_price"
                }, inplace=True)
        data.index.names = ["k"]
        
        tau = (T-t).days/365
        vols = np.array(data["iv"])/100
        strikes =  np.array(data["iv"].index)
        curves.append(xr.DataArray(data = vols.reshape(1, len(vols)), 
            dims=["tau", "k"], 
            coords = {"tau" : [tau], "k": strikes}))
    
    surface = xr.concat(curves, dim="tau").sortby(["tau", "k"])  
    surface.attrs["date"] = t
    surface.attrs["base_price"] = base_price
       
    return surface
