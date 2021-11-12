#Parameter:
product='reanalysis-era5-single-levels-monthly-means'
parameters=['100m_u_component_of_wind',
            '100m_v_component_of_wind',
            ]
#region
north=47
south=34
east=18
west=-7
#time
from_year=1979
to_year=2021
from_month=1
to_month=12

if __name__ == '__main__':
    
    import cdsapi
    import xarray as xr
    c = cdsapi.Client()

    years=[]
    for year in range(from_year, to_year+1):
        years.append(str(year))
    months=[]
    for month in range(from_month, to_month+1):
        if month <= 9:
            months.append('0'+str(month))
        else:
            months.append(str(month))
        
    c.retrieve(product,
        {
            'format': 'netcdf',
            'product_type': 'monthly_averaged_reanalysis',
            'variable': parameters,
            'year': years,
            'month': months,
            'time': '00:00',
            'area': [
                north, west, south, east,
            ],
        },
        str(from_year)+'_'+str(to_year)+'.nc')


