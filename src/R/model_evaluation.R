### Trying to do intersection over union of tree crowns
### this works in matching crowns...but I want matched crowns with IoU > 0.5
### achieved

### now need to crop manual crowns to this area. Then do it...
### And I should be able to count tps, fps, tns and fns from that. DONE.

# Now trying to do just for large trees to extract accuracy and so on for LARGE trees
# Gonna look like this:


rm(list=ls())

#install.packages('mapview')
#install_github('swinersha/UAVforestR')

library(devtools)
library(tidyverse)
library(magrittr)
library(readxl)
library(raster)
library(maptools)
#library(mapview)
library(rgdal)
library(ggplot2)
library(rgeos)
library(foreach)
library(doParallel)
library(sf)
library(UAVforestR)
library(quantreg)

# Load R source files
R_source_files<-list.files(
  path = "R",
  pattern = "*.R$",
  full.names = TRUE
)
sapply(R_source_files, function(x) source(x, local = FALSE,  echo = FALSE))

#######


crown_overlap<-function(auto_trees, manual_trees, buffer_by, verbose='off'){
  # out<-matrix(0, nrow=nrow(manual_trees), ncol=4)
  out<-matrix(0, nrow=nrow(manual_trees), ncol=6)
  out[,1]<-1:nrow(manual_trees)
  out[,3]<-1
  
  auto_trees$id<-1:nrow(auto_trees)
  sum(sapply(auto_trees@polygons, function(x) x@area)<0.001)
  
  for(i in 1:nrow(manual_trees)){
    i_esc<<-i
    #    print(i)
    poly<-manual_trees[i,] # selects the current polygon
    poly_area<-poly@polygons[[1]]@Polygons[[1]]@area # the area of the polygon
    poly_buffer<-gBuffer(poly, width=buffer_by) # makes a buffer around it
    #cropped_trees<-raster::crop(auto_trees, poly_buffer) # crops the auto trees to the buffer
    cropped_trees<-raster::crop(auto_trees, poly) # crops the auto trees to any that intersect with poly
    cropped_trees<-auto_trees[auto_trees$id %in% cropped_trees$id,]
    cropped_trees <- gBuffer(cropped_trees, byid=TRUE, width=0) # deals with self-intersection
    if(!is.null(cropped_trees)){
      for (j in 1:nrow(cropped_trees)) {
        #      print(j)
        overlap <- gIntersection(poly, cropped_trees[j, ])# extracts the intersecting area.
        overlap_area <- overlap@polygons[[1]]@Polygons[[1]]@area # the area of the intersection
        union1 <- gUnion(poly, cropped_trees[j, ])
        union1_area <- union1@polygons[[1]]@Polygons[[1]]@area
        iou <- overlap_area/union1_area
        if (iou > 0.49) {
          auto_area <- cropped_trees@polygons[[j]]@Polygons[[1]]@area
          overlap_area <- overlap@polygons[[1]]@Polygons[[1]]@area # the area of the intersection
          
          # now need to try and work out the metrics - precision/recall/F1
          
          
          #overseg <- (auto_area - overlap_area) / poly_area
          overseg <- 1-(overlap_area / auto_area) # false positive rate
          underseg <- (poly_area - overlap_area) / poly_area # false negative rate
          # overlap_percent<-overlap_area/poly_area # the percentage area
          size_ratio<-auto_area/poly_area # the percentage area
          
          if (out[i, 3] == 1) {
            out[i, 2] <- cropped_trees$id[j]
            out[i, 3] <- 0
            out[i, 4] <- overseg
            out[i, 5] <- underseg
            out[i, 6] <- size_ratio
          }
          else if ((overseg + underseg) < sum(out[i, 4:5])) {
            # stores the result
            out[i, 2] <- cropped_trees$id[j]
            out[i, 4] <- overseg
            out[i, 5] <- underseg
            out[i, 6] <- size_ratio
          }
          # if(verbose=='on')
          #   cat('j: ', j, 'overlap: ', overlap_percent, '\n')
          # if(overlap_percent>out[i,3]){ # stores the result
          #   out[i,1]<-i
          #   out[i,2]<-cropped_trees$id[j]
          #   out[i,3]<-overlap_percent
          #   out[i,4]<-size_ratio
          # }
        }
      }
      if (verbose == 'on')
        cat('out: ', out[i, ], '\n')
    }
  }
  out[out[,2]==0,2]<-NA
  # Loads these as additional columns for the manual trees:
  manual_trees$id_auto_tree<-out[,2]
  # manual_trees$overlap_auto_tree <- out[, 3]
  # manual_trees$size_ratio <- out[, 4]
  manual_trees$tree_match <- out[, 3]
  manual_trees$overseg <- out[, 4]
  manual_trees$underseg <- out[, 5]
  manual_trees$cost <- rowSums(out[,4:5])/2 + out[,3]
  manual_trees$size_ratio<-out[,6]
  
  if (verbose == 'on')
    cat('Cost: ', manual_trees$cost, '\n')
  
  return(manual_trees)
}

#####
#manual<-readOGR("C:/Users/sebhi/ai4er/mres_project/work/data/paracou_data/crowns/all_combined_crowns.shp")

### these are the PARACOU manuals
#manual<-readOGR("C:/Users/sebhi/ai4er/mres_project/work/data/manual_crowns/cropped_paracou_plots/fresh_2019_10cm_crowns_286555_583760.shp")
#manual<-readOGR("C:/Users/sebhi/ai4er/mres_project/work/data/manual_crowns/cropped_paracou_plots/fresh_2019_10cm_crowns_286995_583630.shp")
#manual<-readOGR("C:/Users/sebhi/ai4er/mres_project/work/data/manual_crowns/cropped_paracou_plots/fresh_2019_10cm_crowns_287185_584065.shp")

### These are the SEPILOK manuals
#manual<-readOGR("C:/Users/sebhi/ai4er/mres_project/work/data/manual_crowns/eval_sepilok_plots/sepilok_603330_647470_manuals.shp")
#manual<-readOGR("C:/Users/sebhi/ai4er/mres_project/work/data/manual_crowns/eval_sepilok_plots/sepilok_603360_647800_manuals.shp")
manual<-readOGR("C:/Users/sebhi/ai4er/mres_project/work/data/manual_crowns/eval_sepilok_plots/sepilok_603450_648000_manuals.shp")



### RUN this line if you want to assess on the large crowns
#manual_large = manual[sapply(manual@polygons, function(x) x@area>225),]


#man_crowns<-readOGR('data/shape/ITC_trees_params_cost_uav/Matched_costed/seed_0.6_crown_0.7_sobel_5.5.shp')
#ut<-readOGR("C:/Users/sebhi/ai4er/mres_project/work/data/detectron_predicted_crowns/286555_583760-preds_ten_percent_conf.shp")

### these are the good PARACOU uts for 20% confidence
#ut<-readOGR("C:/Users/sebhi/ai4er/mres_project/work/data/detectron_predicted_crowns/286555_583760_20_conf.shp")
#ut<-readOGR("C:/Users/sebhi/ai4er/mres_project/work/data/detectron_predicted_crowns/286995_583630_20_conf.shp")
#ut<-readOGR("C:/Users/sebhi/ai4er/mres_project/work/data/detectron_predicted_crowns/287185_584065_20_conf.shp")

### maybe need Paracou here at 10% confidence...

### these PARACOU uts are at 10% confidence
#ut<-readOGR("C:/Users/sebhi/ai4er/mres_project/work/data/detectron_predicted_crowns/286555_583760_10_conf.shp")
#ut<-readOGR("C:/Users/sebhi/ai4er/mres_project/work/data/detectron_predicted_crowns/286995_583630_10_conf.shp")
#ut<-readOGR("C:/Users/sebhi/ai4er/mres_project/work/data/detectron_predicted_crowns/287185_584065_10_conf.shp")
#ut<-readOGR("C:/Users/sebhi/ai4er/mres_project/work/data/detectron_predicted_crowns/testing2.shp")

### these SEPILOK uts are at 10% confidence
#ut<-readOGR("C:/Users/sebhi/ai4er/mres_project/work/data/detectron_predicted_crowns/sepilok_603330_647470.shp")
#ut<-readOGR("C:/Users/sebhi/ai4er/mres_project/work/data/detectron_predicted_crowns/sepilok_603360_647800.shp")
ut<-readOGR("C:/Users/sebhi/ai4er/mres_project/work/data/detectron_predicted_crowns/sepilok_603450_648000.shp")
ut = ut[ut$pixelvalue>100,] 
#ut = ut[ut$perimeter<300,]
plot(ut)

### RUN for assessing on large crowns
#ut_large = ut[sapply(ut@polygons, function(x) x@area>225),]
#plot(ut_large)

# ignore these uts for now
#ut<-readOGR("C:/Users/sebhi/ai4er/mres_project/work/data/detectron_predicted_crowns/286555_583760_preds_merged_shape_with_box.shp")
#ut<-readOGR("C:/Users/sebhi/ai4er/mres_project/work/data/detectron_predicted_crowns/shapefile_of_pred_wout_box_010721.shp")


crs(ut)<-crs(manual)
crs(ut_large)<-crs(manual_large)
#crs(manual)<-crs(ut)
# Find the 'correct' crowns with IoU > 0.5:
matched<-crown_overlap(auto_trees=ut, manual_trees=manual, buffer_by=0)
#matched<-crown_overlap(auto_trees=ut_large, manual_trees=manual_large, buffer_by=0)

out_tmp<-data.frame(manual.id=i,
                    n.unmatch = sum(matched$tree_match),
                    mean.over = mean(matched$overseg),
                    sd.over = sd(matched$overseg),
                    mean.under = mean(matched$underseg),
                    sd.under = sd(matched$underseg),
                    mean.cost = mean(matched$cost),
                    med.cost = median(matched$cost),
                    sd.cost = sd(matched$cost),
                    mean.size_ratio = mean(matched$size_ratio),
                    med.size_ratio = median(matched$size_ratio),
                    sd.size_ratio = sd(matched$size_ratio)
)

out[[i]]<-out_tmp
# add the auto tree data to the manual trees:

##### need to change back to not large trees

no_of_preds <- length(ut)
no_of_manuals <- length(manual)
matched<-matched[!is.na(matched$id_auto_tree),]
auto<-ut[matched$id_auto_tree,]
#plot(matched)
#plot(auto)
#length(matched)
#length(auto)
tp = length(matched)
fp = no_of_preds - tp
fn = no_of_manuals - length(matched)

# check for false positives, as this measure is not very good, since currently I can't filter the uts by area.

precision = tp/(tp+fp)
recall = tp/(tp + fn)
fscore = (2*precision*recall)/(precision+recall)

matched<-matched[!is.na(matched$id_auto_tree),]
auto<-ut[matched$id_auto_tree,]
auto@data$shpbnd<-rowSums(auto@data[,c('hbnd_mn', 'hbnd_mx', 'soblbnd')])
auto@data$spcbnd<-rowSums(auto@data[,c('allmbnd', 'crwnbnd')])
auto@data<-auto@data[,c('trHghts_mn', 'trHghts_mx', 'R', 'shpbnd', 'spcbnd', 'sobl_mn')]
matched@data<-cbind(matched@data, auto@data)

# par(mfrow=c(1,1), mar=c(0,0,0,0))
length(ut)
plot(manual)
plot(ut, add=TRUE, border='red')
plot(auto, add=TRUE, border='blue')

# Save the output:
### save it somehow here...
