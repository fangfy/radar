
import glob, os
from datacube import helpers
from datacube.utils import geometry
import xarray as xr
import subprocess


def convert_to_tiff(filename, var=None, outputdir = 'geotiff'):
    if var is None:
        ds =  xr.open_dataset(filename)
        varnames = list(ds.data_vars)
        if None in varnames:
            raise ValueError(varnames)
    else:
        ds =  xr.open_dataset(filename)
        varnames = [var]
    for var in varnames:
        outputname = '%s/%s'%(outputdir, filename.split('/')[-1].replace('.nc','_%s.tif'%var.lower()))
        if os.path.exists(outputname): continue
    #print(outputname)
        try:
            ds_output = ds[var].to_dataset(name=var)
        except:
            print(ds.data_vars)
        #ds_output = ds_output.sortby('y', ascending=False)
    #ds = ds.astype('float64')
        ds_output.attrs['crs'] = geometry.CRS('EPSG:3577')
    #print(ds)
        helpers.write_geotiff(outputname, ds_output)
    return varnames


filenames = glob.glob('s1_median/*.nc')
outputdir = 's1_median_geotiff'

for filename in filenames:
    varnames = convert_to_tiff(filename, outputdir = outputdir)
 
for var in varnames:
    vrtname = 's1_median.vrt'
    #if not os.path.exists(vrtname):
    cmd = 'gdalbuildvrt %s %s/*_%s.tif'%(vrtname, outputdir, var.lower())
    subprocess.call(cmd, shell=True)
