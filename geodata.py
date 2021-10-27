# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 22:02:22 2021

@author: hw9335
"""

from pyrosm import get_data
from pyrosm import OSM


# fp = get_data("data/texas-latest.osm.pbf")


fp = get_data("texas", directory="data")
#fp = get_data("Helsinki", directory="data")

osm = OSM(fp)

drive_net = osm.get_network(network_type="driving")
drive_net.plot()

boundaries = osm.get_boundaries()
