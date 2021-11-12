# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 14:11:25 2021

@author: thiba
"""
import netCDF4 as nc
import pandas as pd
import xarray as xr
import numpy as np
import csv


import streamlit as st
import netCDF4
import pandas as pd
import matplotlib.pyplot as plt
import folium
import mplleaflet
import numpy as np
import branca
from folium import plugins
from scipy.interpolate import griddata
import geojsoncontour
import scipy as sp
import scipy.ndimage
from streamlit_folium import folium_static


st.sidebar.title('Projet "Meteo Wind"')
st.sidebar.subheader('Menu')
dif_parti=["Carte des vents","Evolution du vent en un point", "Bathymétrie", \
           "Faune protégée",'Flore protégée','Couloir de migration des petits oiseaux']
Partie=st.sidebar.radio(' ',options=dif_parti)

from_year = st.sidebar.number_input("A partir de l'année", value=2020)
to_year = st.sidebar.number_input("Jusqu'à de l'année", value=2020)
north = st.sidebar.number_input('Latitude Nord', value=44.0)
south = st.sidebar.number_input('Latitude Sud', value=42.0)
east = st.sidebar.number_input('Longitude Est', value=8.0)
west = st.sidebar.number_input('Longitude Ouest', value=6.0)


#Parameter:
product='reanalysis-era5-single-levels-monthly-means'
parameters=['100m_u_component_of_wind',
            '100m_v_component_of_wind',
            ]

from_month=3
to_month=3

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


#-------Extration des données et stockage que format csv-----------#
# Transformation des données en dataframe mais cette solution est mis de coté pour le moment 
#ds0 = xr.open_dataset(fn)
#df = ds0.to_dataframe()
# stockage que format csv
def ncdfToCsv(ncdffile, outputfilename):
  """
  :param ncdffile: path to data in ncdf format. The file is from cds data producer
  :type ncdffile: string
  """
  
  ds = nc.Dataset(ncdffile)

  lats = ds.variables['latitude'][:]  
  lons = ds.variables['longitude'][:]
  time = ds.variables['time'][:]
  v_wind = ds.variables['v100'][:]
  u_wind = ds.variables['u100'][:] 

  np_lats = np.ones((lats.shape[0], lons.shape[0]))
  np_lons = np.ones((lats.shape[0], lons.shape[0]))

  i=0
  
  for (lat)  in (lats):
      #print(lat)
      np_lats[i, :] = lat;
      #print(np_lats)
      i+= 1
  i=0
  for (lon)  in (lons):
      #print(lon)
      np_lons[:, i] = lon
      #print(np_lons)
      i+= 1
  
  with open(outputfilename, 'w', encoding='UTF8') as f:
      writer = csv.writer(f)
      header = ['time', 'lon', 'lat', 'v', 'u']
      # write the header
      writer.writerow(header)
      #file file with data
      num_mois=0
      
      for mois in time:
          V = v_wind[num_mois]
          U = u_wind[num_mois]
          for num_lat in range(np_lats.shape[0]):
              for num_lon in range(np_lons.shape[1]):
                  lon = np_lons[num_lat, num_lon]
                  lat = np_lats[num_lat, num_lon]
                  
                  v = V[num_lat, num_lon]
                  u = U[num_lat, num_lon]
                  if (u!="--" or v!="--"):
                      #print(u, v)
                      # write the data
                      data = [mois, lon, lat, v, u]
                      writer.writerow(data)
    
          num_mois +=1;


ncdfToCsv(str(from_year)+'_'+str(to_year)+'.nc', str(from_year)+'_'+str(to_year)+'.csv')
df=pd.read_csv(str(from_year)+'_'+str(to_year)+'.csv')

#df=pd.read_csv('data_tibaut.csv')
df = df[df["u"] != "--"] #on élimine les '--' contenus sur certaines lignes
df["u"] = df["u"].astype(float)
df["v"] = df["v"].astype(float)
df["Vitesse"] = round((df["u"]**2 + df["v"]**2)**0.5,1)


if Partie==dif_parti[0]:
    st.title("Carte des vents")
    st.info("Cette carte permet de repérer les sites où la vitesse du vent est la plus élevée sur la période et sur la fenêtre géographique choisies par l'utilisateur")
    
    # latitude_sup = st.number_input('Latitude supérieure', value=44.0)
    # latitude_inf = st.number_input('Latitude inférieure', value=42.0)
    # longitude_sup = st.number_input('Longitude supérieure', value=8.0)
    # longitude_inf = st.number_input('Longitude inférieure', value=6.0)
    
    df = df[(df["lat"] <= north) & (df["lat"] >= south) &\
        (df["lon"] <= east) & (df["lon"] >= west)]
        
        

    x = df["lon"]
    y = df["lat"]
    U = df["u"]
    V = df["v"]
    fig, ax = plt.subplots()
    
    kw = dict(color='black', alpha=0.8, scale=50)
    q = ax.quiver(x, y, U, V, **kw)
    
    gj = mplleaflet.fig_to_geojson(fig=fig)
    
    
    feature_group0 = folium.FeatureGroup(name='Vecteurs')
    #feature_group1 = folium.FeatureGroup(name='coucou')  
    
    
    mapa = folium.Map(location=[y.mean(), x.mean()], tiles="Stamen Terrain",
                      zoom_start=7)
    
    for feature in gj['features']:
        if feature['geometry']['type'] == 'Point':
            lon, lat = feature['geometry']['coordinates']
            div = feature['properties']['html']
    
            icon_anchor = (feature['properties']['anchor_x'],
                           feature['properties']['anchor_y'])
    
            icon = folium.features.DivIcon(html=div,
                                           icon_anchor=icon_anchor)
            marker = folium.Marker([lat, lon], icon=icon)
            feature_group0.add_child(marker)
        else:
            msg = "Unexpected geometry {}".format
            raise ValueError(msg(feature['geometry']))
            
    mapa.add_child(feature_group0)
    #mapa.add_children(feature_group1)
    
    
    for i in range(df.shape[0]):
        latitude = df.iloc[i,2]
        longitude = df.iloc[i,1]
        vitesse = df.iloc[i,5]
        m = "<strong>" + "Vitesse : " + "</strong>" + str(vitesse) + " m/s" + \
            "<br><strong>" + "Point : " + "</strong>" + str(i)
        
        folium.CircleMarker([latitude, longitude], radius = 5,color=None,fill_color ="red",
                        fill_opacity=0.5,popup = folium.Popup(m, max_width = 400)).add_to(mapa)
    
        
    
    
    
    # Setup colormap
    colors = ["#ffeda0" ,"#fed976", "#feb24c", "#fd8d3c", "#fc4e2a", "#e31a1c", "#bd0026", "#800026"]
    vmin   = 0 
    vmax   = df["Vitesse"].max()
    levels = len(colors)
    feature_group1 = branca.colormap.LinearColormap(colors, vmin=vmin, vmax=vmax).to_step(levels)
     
    # The original data
    x_orig = np.asarray(df.lon.tolist())
    y_orig = np.asarray(df.lat.tolist())
    z_orig = np.asarray(df.Vitesse.tolist())
     
    # Make a grid
    x_arr          = np.linspace(np.min(x_orig), np.max(x_orig), 500)
    y_arr          = np.linspace(np.min(y_orig), np.max(y_orig), 500)
    x_mesh, y_mesh = np.meshgrid(x_arr, y_arr)
     
    # Grid the values
    z_mesh = griddata((x_orig, y_orig), z_orig, (x_mesh, y_mesh), method='linear')
     
    # Gaussian filter the grid to make it smoother
    sigma = [5, 5]
    z_mesh = sp.ndimage.filters.gaussian_filter(z_mesh, sigma, mode='constant')
     
    # Create the contour
    contourf = plt.contourf(x_mesh, y_mesh, z_mesh, levels, alpha=0.5, colors=colors, linestyles='None', vmin=vmin, vmax=vmax)
     
    # Convert matplotlib contourf to geojson
    geojson = geojsoncontour.contourf_to_geojson(
        contourf=contourf,
        min_angle_deg=3.0,
        ndigits=5,
        stroke_width=1,
        fill_opacity=0.9);
     
    # Plot the contour plot on folium
    folium.GeoJson(
        geojson,
        style_function=lambda x: {
            'color':     x['properties']['stroke'],
            'weight':    x['properties']['stroke-width'],
            'fillColor': x['properties']['fill'],
            'opacity':   1,
        }, name="Couleurs d'interpolation").add_to(mapa)
     
    # Add the colormap to the folium map
    feature_group1.caption = 'Wind speed'
     
    
    mapa.add_child(feature_group1)
    mapa.add_child(folium.map.LayerControl())
    
    folium_static(mapa)
    
if Partie==dif_parti[1]:
    #Parameter:
    product='reanalysis-era5-single-levels-monthly-means'
    parameters=['100m_u_component_of_wind',
                '100m_v_component_of_wind',
                ]
    
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
    def ncdfToCsv(ncdffile, outputfilename):
      """
      :param ncdffile: path to data in ncdf format. The file is from cds data producer
      :type ncdffile: string
      """
      
      ds = nc.Dataset(ncdffile)
    
      lats = ds.variables['latitude'][:]  
      lons = ds.variables['longitude'][:]
      time = ds.variables['time'][:]
      v_wind = ds.variables['v100'][:]
      u_wind = ds.variables['u100'][:] 
    
      np_lats = np.ones((lats.shape[0], lons.shape[0]))
      np_lons = np.ones((lats.shape[0], lons.shape[0]))
    
      i=0
      
      for (lat)  in (lats):
          #print(lat)
          np_lats[i, :] = lat;
          #print(np_lats)
          i+= 1
      i=0
      for (lon)  in (lons):
          #print(lon)
          np_lons[:, i] = lon
          #print(np_lons)
          i+= 1
      
      with open(outputfilename, 'w', encoding='UTF8') as f:
          writer = csv.writer(f)
          header = ['time', 'lon', 'lat', 'v', 'u']
          # write the header
          writer.writerow(header)
          #file file with data
          num_mois=0
          
          for mois in time:
              V = v_wind[num_mois]
              U = u_wind[num_mois]
              for num_lat in range(np_lats.shape[0]):
                  for num_lon in range(np_lons.shape[1]):
                      lon = np_lons[num_lat, num_lon]
                      lat = np_lats[num_lat, num_lon]
                      
                      v = V[num_lat, num_lon]
                      u = U[num_lat, num_lon]
                      if (u!="--" or v!="--"):
                          #print(u, v)
                          # write the data
                          data = [mois, lon, lat, v, u]
                          writer.writerow(data)
        
              num_mois +=1;
    
    
    ncdfToCsv(str(from_year)+'_'+str(to_year)+'.nc', str(from_year)+'_'+str(to_year)+'.csv')
        
        
    st.title("Evolution mensuelle de la vitesse moyenne du vent")
    st.info("Après avoir repéré un point à fort potentiel, l'utilisateur peut en voir l'évolution sur une longue durée")
    
    def GetSpeedTimeForEachPoint0(ncdffile='1978_2021.nc'):
      ds = nc.Dataset(ncdffile)
      lats = ds.variables['latitude'][:]  
      lons = ds.variables['longitude'][:]
      time = ds.variables['time'][:]
      v_wind = ds.variables['v100'][:]
      u_wind = ds.variables['u100'][:] 
    
      np_lats = np.ones((lats.shape[0], lons.shape[0]))
      np_lons = np.ones((lats.shape[0], lons.shape[0]))
    
      i=0
      for (lat)  in (lats):
        #print(lat)
        np_lats[i, :] = lat;
        #print(np_lats)
        i+= 1
      i=0
      for (lon)  in (lons):
        #print(lon)
        np_lons[:, i] = lon
        #print(np_lons)
        i+= 1
    
      np_speed = np.ones((np_lons.shape[0] * np_lons.shape[1], time.shape[0]))
      np_time = np.ones((np_lons.shape[0] * np_lons.shape[1], time.shape[0]))
      a = 0
      for num_lat in range(np_lats.shape[0]):
        for num_lon in range(np_lons.shape[1]):
          for num_mois in range(time.shape[0]):
            V = v_wind[num_mois][0]
            U = u_wind[num_mois][0]
            v = V[num_lat, num_lon]
            u = U[num_lat, num_lon]
            np_speed[a, num_mois] = (u**2 + v**2)**0.5
            np_time[a, num_mois] = num_mois
          a +=1
        a = 0
      return(np_speed, np_time)
    num_pt = st.number_input('Numéro du point :', value=6)
  
    def GetSpeedTimeForEachPoint(cscdffile, coord):
        df=pd.read_csv(cscdffile)
        df = df[(df["lon"]==coord[0]) & (df["lat"]==coord[1])]
        df["Vitesse"] = round((df["u"]**2 + df["v"]**2)**0.5,1)
        
        return(df["time"].to_numpy(), df["Vitesse"].to_numpy())
    
    time,  v = GetSpeedTimeForEachPoint(str(from_year)+'_'+str(to_year)+'.csv', [df["lon"][num_pt], df["lat"][num_pt]])
    
    
    fig = plt.figure(figsize=(8, 6))
    plt.xlabel("Time in months")
    plt.ylabel("Wind speed (m/s)")
    plt.plot(np.arange(0,len(time)),v)
    
    st.pyplot(fig);
    
if Partie==dif_parti[2]:
    st.title("Bathymétrie")
    st.info("Cette carte permettra de croiser les informations de fonds marins avant d'y implanter une éolienne")
    