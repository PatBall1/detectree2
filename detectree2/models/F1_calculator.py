# necessary imports 
import os 
import numpy as np
import json
import type
from shapely.geometry import shape, Polygon

def PolyArea(feature= type(dict)):
  """
  Take a featrue from a geojson and calculates the area of the polygon
  """
  coordinates = feature["geometry"]["coordinates"][0]
  coord_tuples=[]

  for entry in coordinates:
    coord_tuples.append((entry[0], entry[1]))

  polygon = Polygon(coord_tuples)

  return polygon.area

def all_polys_area(features = list):
  """
  Take a list of polygons and find the corresponding area of each of them
  """

  poly_areas = {}
  poly_count = 0
  for feat in features:
    poly_areas[str(poly_count)]=PolyArea(feat)
    poly_count +=1

  return poly_areas

def intersection_data(test_features =list,
                       pred_features =list,
                       test_feats_areas = type(dict), 
                       pred_feats_areas = type(dict)
                           ):
  """
  Generates a dictionary of the intersections and IoU for each tile
  """
  # Create a list of the test features appearing so we can caluclate the number of false negatives easier
  test_feats_appearing = [] 
  
  all_tile_intersections = []
  pred_count = 0
  for pred_feat in pred_features:
    test_count = 0
    for test_feat in test_features:
      if shape(test_feat['geometry']).intersects(shape(pred_feat['geometry'])):
        intersection = {}
        intersection["pred_feat_"+ str(pred_count)+"_area"]= pred_feats_areas[str(pred_count)]
        intersection["test_feat_"+ str(test_count)+"_area"]= test_feats_areas[str(test_count)]
        test_feats_appearing.append(test_count)
        intersection["Intersection"] = (shape(pred_feat['geometry']).intersection(shape(test_feat['geometry']))).area
        
        union_area = test_feats_areas[str(test_count)]+ pred_feats_areas[str(pred_count)] -  intersection["Intersection"]
        intersection["IoU"] = intersection["Intersection"] / union_area


        all_tile_intersections.append(intersection)

      # increase test count outside the if statement for it to work
      test_count +=1

    # increase the pred count so the objects are labelled correctly
    pred_count +=1

    false_negatives = test_count - len(set(test_feats_appearing))
  
  return all_tile_intersections, false_negatives

def threshold_positives(
    tile_intersection = list,
    num_test_feats = int,
    threshold = 0.5
    ):
  
  true_positives = 0
  false_positives = 0
  false_negatives = 0

  for entry in tile_intersection:
    if entry["IoU"] >= threshold:
      true_positives +=1
    else: 
      false_positives +=1
  
  return true_positives, false_positives


def prec_recall_func(
    total_tps, 
    total_fps, 
    total_fns):
  "Calculate the precision and recall by standard formulas"

  precision = total_tps/(total_tps+total_fps)
  recall = total_tps/(total_tps+total_fns)
  
  return precision, recall

def f1_cal(precision, recall):
  "Calculating the F1 score"

  return (2*precision*recall)/(precision+recall)

def site_F1_score(
    test_directory=None,
    pred_directory=None,
    EPSG = None
    ):
  """
  Code to calculate all the intersections of shapes in a pair of files and the area of the corresponding polygons
  Output the test_count so 
  """

  if EPSG == None:
    raise ValueError('Set the EPSG value')

  test_entries = os.listdir(test_directory)
  site_intersections={}
  total_tps = 0
  total_fps = 0
  total_fns = 0

  for file in test_entries:
    if ".geojson" in file:
      
      #open the geojson in the test folder and the corresponding one in the prediction folder
      with open(test_directory+file) as test_file:
        test_json = json.load(test_file)
      test_features = test_json["features"]
      
      pred_file_path = pred_directory + "Prediction_"+ file.replace('.geojson', '_'+ EPSG + '.geojson')
      with open(pred_file_path) as pred_file:
        pred_json = json.load(pred_file)
      pred_features = pred_json["features"]

      # create a dict of all intersections and their area
      test_feats_areas = all_polys_area(test_features)
      test_feat_count = len(test_feats_areas)
      pred_feats_areas = all_polys_area(pred_features)
      pred_feat_count = len(pred_feats_areas)

      # print(test_feats_areas)
      # print(pred_feats_areas)
      
      tile_intersections, fns = intersection_data(test_features, pred_features, test_feats_areas, pred_feats_areas)
      tps, fps = threshold_positives(tile_intersections)
      
      # update the information
      site_intersections[file] = tile_intersections
      total_tps = total_tps + tps
      total_fps = total_fps + fps
      total_fns = total_fns + fns

  prec, rec = prec_recall_func(total_tps, total_fps, total_fns)
  f1_score = f1_cal(prec, rec)

  print(f1_score)
  

