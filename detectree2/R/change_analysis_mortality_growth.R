# clear the environment

rm(list=ls())

# it is possible that this one has to be imported first

library(velox)

library(raster)
library(dplyr)
library(tidyr)
library(magrittr)
library(ggplot2)
library(RColorBrewer)
library(rgdal)
library(MASS)

# install.packages("ggpointdensity")
library(ggpointdensity)

### Function that we are going to need later on
### I think this going to need some editing...

get_poly_CHM_info_velox=function(rpoly,CHM_org,CHM_2020,CHM_diff){
  
  CHM_orgv=velox(CHM_org)
  CHM_2020v=velox(CHM_2020)
  CHM_diffv=velox(CHM_diff)
  
  # Get info from original raster
  # does this introduce NaNs?
  poly_original_raster_data_list= CHM_orgv$extract(rpoly, fun = NULL)
  rpoly$area=sapply(poly_original_raster_data_list,FUN=length) # poly area in raster units
  rpoly$Org_H_md=sapply(poly_original_raster_data_list,FUN=median,na.rm=TRUE)
  rpoly$Org_H_mean=sapply(poly_original_raster_data_list,FUN=mean,na.rm=TRUE)
  rpoly$Org_H_max =sapply(poly_original_raster_data_list,FUN=max)
  rpoly$Org_H_min =sapply(poly_original_raster_data_list,FUN=min)
  rpoly$Org_H_var =sapply(poly_original_raster_data_list,FUN=var)
  
  #poly_2020_raster_data_list= CHM_2020v$extract(rpoly, fun = NULL)
  #rpoly$area=sapply(poly_2020_raster_data_list,FUN=length) # poly area in raster units
  #rpoly$H_mean_2020=sapply(poly_2020_raster_data_list,FUN=mean,na.rm=TRUE)
  #rpoly$H_max_2020 =sapply(poly_2020_raster_data_list,FUN=max)
  #rpoly$H_min_2020 =sapply(poly_2020_raster_data_list,FUN=min)
  #rpoly$H_var_2020 =sapply(poly_2020_raster_data_list,FUN=var)
  
  poly_change_raster_data_list=CHM_diffv$extract(rpoly, fun = NULL)
  rpoly$Chng_H_md=sapply(poly_change_raster_data_list,FUN=median,na.rm=TRUE)
  rpoly$Change_H_mean=sapply(poly_change_raster_data_list,FUN=mean)
  rpoly$Change_H_max =sapply(poly_change_raster_data_list,FUN=max)
  rpoly$Change_H_min =sapply(poly_change_raster_data_list,FUN=min)
  rpoly$Change_H_var =sapply(poly_change_raster_data_list,FUN=var)
  
  #rpoly$perimeter=as.numeric(polyPerimeter(rpoly)) # perimeter of each polygon
  #rpoly$shape_complexity = as.numeric(rpoly$perimeter/(2*sqrt(rpoly$area*pi)))
  #rpoly$shape_circleness=as.numeric(4*pi*(rpoly$area)/((rpoly$perimeter)^2))
  return(rpoly)
}


######################################################################
# MAKING GENERAL PLOTS FOR RESOLUTION AND CROSS SITE COMPARISON ######
######################################################################

# dataset of F1-scores with tree height lower bounds
tree_h_lower_bound <- c(20, 30, 40, 50, 60)
f1_score <- c(0.62, 0.64, 0.68, 0.78, 0.85)

df <- data.frame(tree_h_lower_bound, f1_score)

print(df)

#plot all three series on the same chart using geom_line()
ggplot(data = df, aes(x=tree_h_lower_bound, y=f1_score))+
  geom_point(aes(x=tree_h_lower_bound, y=f1_score), alpha = 1, color='skyblue')+
  geom_line(color='skyblue')+
  ggtitle('Average F1 Score for trees of increasing height')+
  theme_classic()+
  theme(plot.title=element_text(size=10,face='bold',margin=margin(10,10,10,0), hjust = 0.5))+
  labs(x='Tree Height / m', y='F1 Score')+
  theme(legend.title = element_text(colour="black", size=8, face="bold"))+
  labs(size='Crown Area')+
  guides(size = guide_legend(reverse=TRUE))
  
ggsave("C:/Users/sebhi/ai4er/tree_height_f1.png")
  
# data for training and testing on different resolutions
testing_res <- c(0.1, 0.5, 1, 2)
train_test_new_res <- c(60.63, 59.55, 58.47, 51.54)
test_new_res <- c(60.63, 57.26, 53.00, 33.84)

df <- data.frame(testing_res, train_test_new_res, test_new_res)
data_long <- melt(df, id = "testing_res")

print(data_long)

#plot all three series on the same chart using geom_line()

ggplot(data = data_long, aes(x=testing_res, y=value, color= variable))+
  geom_point()+
  geom_line()+
  ggtitle('Model Performance on Different Resolutions')+
  theme_classic()+
  theme(plot.title=element_text(size=10,face='bold',margin=margin(10,10,10,0), hjust = 0.5))+
  labs(x='Image resolution / m2', y='AP50')+
  theme(legend.title = element_text(colour="black", size=8, face="bold"))+
  #guides(size = guide_legend(reverse=TRUE))+
  #labs(color = "Mode of Train and Test")+
  scale_colour_discrete(name  ="Mode of Test and Train", 
                        breaks=c("train_test_new_res", "test_new_res"),
                        labels=c("Train and Test", "Test"))

ggsave("C:/Users/sebhi/ai4er/resolution_testing.png")

# cross-site performance
type_of_forest <- c('Trained on Different Forest', 'Trained on Similar Forest', 'Trained on Same Forest')
danum <- c(31, 46, 51.54)
paracou <- c(48.3, 51.3, 53.0)
sepilok_east <- c(48.1, 50.2, 36)
sepilok_west <- c(38.9, 51.26, 54.3)

df <- data.frame(type_of_forest, danum, paracou, sepilok_east, sepilok_west)
data_long <- melt(df, id = "type_of_forest")

print(data_long)

#plot all three series on the same chart using geom_line()


ggplot(data = data_long, aes(x=type_of_forest, y = value, color = variable))+
  geom_point()+
  geom_line()+
  ggtitle('Model Performance across Sites')+
  theme_classic()+
  theme(plot.title=element_text(size=10,face='bold',margin=margin(10,10,10,0), hjust = 0.5))+
  labs(x='', y='AP50')+
  theme(legend.title = element_text(colour="black", size=8, face="bold"))+
  #guides(size = guide_legend(reverse=TRUE))+
  #labs(color = "Mode of Train and Test")+
  scale_colour_discrete(name  ="Site", 
                        breaks=c("danum", "paracou", "sepilok_east", "sepilok_west"),
                        labels=c("Danum", "Paracou", "Sepilok East", "Sepilok West"))
  
ggsave("C:/Users/sebhi/ai4er/sep_cross_site.png")

# SEPILOK

sep_crowns = shapefile("C:/Users/sebhi/ai4er/mres_project/full_sepilok.shp")

sep_dfs = shapefile("C:/Users/sebhi/ai4er/mres_project/work/final_dfs/sep_dfs.shp")

sep_east_dfs = shapefile("C:/Users/sebhi/ai4er/mres_project/work/final_dfs/sep_east.shp")
sep_east_dfs$Site = "Sepilok East"

sep_west_dfs = shapefile("C:/Users/sebhi/ai4er/mres_project/work/final_dfs/sep_west.shp")
sep_west_dfs$Site = "Sepilok West"
  
# now need to deal with east and west

# Read in the rasters...

Sep_2014 = raster(paste0("C:/Users/sebhi/ai4er/mres_project/work/data/sepilok/sepilok_lidar/Sep_2014_coarse_CHM_g10_sub0.01_0.5m.tif"))
Sep_2020 = raster(paste0("C:/Users/sebhi/ai4er/mres_project/work/data/sepilok/sepilok_lidar/Sep_2020_CHM_g10_sub0.2_0.5m.tif"))
Sep_diff = raster(paste0("C:/Users/sebhi/ai4er/mres_project/work/data/sepilok/sepilok_lidar/Sep_dDSM_random_1m.tif"))

plot(Sep_diff)
plot(sep_dfs, add=TRUE)



# let's have a look at a summary of our crowns
#summary(sep_crowns)

### let's clean up the odd ones...

# drop the NaNs

#sep_crowns_clean = sep_crowns[!is.na(sep_crowns$prmtr), ]

#dfs_sep$perimeter=as.numeric(polyPerimeter(Sep_detectron_crowns))

#sep_crowns_clean = sep_crowns_clean[sep_crowns_clean$prmtr > 5,]
#sep_crowns_clean = sep_crowns_clean[sep_crowns_clean$area < 3000,]
#sep_crowns_clean = sep_crowns_clean[sep_crowns_clean$area > 5,]

#summary(sep_crowns_clean)

### now let us try to do this good business to get changes in height and so forth

# this doesn't work since the raster is too big...so let's
# just read in the dfs created on MAGEOHub and placed in folder lustre_scratch/final_dfs

#sep_dfs = get_poly_CHM_info_velox(rpoly=sep_crowns_clean, CHM_org=Sep_2014, CHM_2020=Sep_2020, CHM_diff=Sep_diff)

#sep_dfs$Site="Sepilok"

summary(sep_dfs)

sep_east_dfs = shapefile("C:/Users/sebhi/ai4er/mres_project/work/final_dfs/sep_east.shp")

sep_east_dfs$area_sqm <- raster::area(sep_east_dfs) 

# let's filter out the trees smaller than 10 metres
sep_east_dfs@data = sep_east_dfs@data[sep_east_dfs@data$Org_H_md > 10,]

# use this clever @data method to get around the problem of NAs in row index

sep_east_dfs@data = sep_east_dfs@data[sep_east_dfs@data$prmtr > 5,]
sep_east_dfs@data = sep_east_dfs@data[sep_east_dfs@data$area_sqm < 3000,]
sep_east_dfs@data = sep_east_dfs@data[sep_east_dfs@data$area_sqm > 10,]
sep_east_dfs$Site = "Sepilok East"

sep_east_dfs = sep_east_dfs[!is.na(sep_east_dfs$ID2),]

file_name<-paste("C:/Users/sebhi/ai4er/mres_project/sep_east_height_drop_data_for_plot")

writeOGR(sep_east_dfs, dsn = paste(file_name, '.shp', sep=''),layer = basename(file_name),drive = 'ESRI Shapefile')

sep_east_dfs = shapefile("C:/Users/sebhi/ai4er/mres_project/work/final_dfs/sep_east.shp")

ggplot(sep_east_dfs@data,aes(Org_H_md, Chng_H_md))+
  geom_point(aes(Org_H_md, Chng_H_md, color=Site), alpha = 0.18)+
  geom_abline(intercept = 0, slope = -0.06, color='navy', alpha = 0.5, linetype='dashed')+
  geom_abline(intercept = 3.4, slope = -0.06, color='navy', alpha = 0.5, linetype='dashed')+
  geom_smooth(method='rlm', aes(Org_H_md, Chng_H_md), size = 0.5, color='red')+
  ggtitle('Determining mortality events from height change')+
  #geom_hline(yintercept=0, linetype='dashed', size=0.1)+
  theme_classic()+
  theme(plot.title=element_text(size=10,face='bold',margin=margin(10,10,10,0), hjust = 0.5))+
  labs(x='2014 tree height / m', y='Change in tree height / m')+
  theme(legend.title = element_text(colour="black", size=8, face="bold"))+
  labs(size='Crown Area')+
  guides(size = guide_legend(reverse=TRUE))+
  xlim(0,65)+
  ylim(-40, 10)


ggsave("C:/Users/sebhi/ai4er/sep_east_growth_bigger.png")

sep_east_fit <- rlm(Chng_H_md ~ Org_H_md, data = sep_east_dfs@data)
summary(sep_east_fit)

sep_east_fit_1sd <- predict(sep_east_fit, interval="prediction", level = 0.68)
sep_east_fit_2sd <- predict(sep_east_fit, interval="prediction", level = 0.95)
sep_east_fit_3sd <- predict(sep_east_fit, interval="prediction", level = 0.997)

summary(sep_east_fit_1sd)
summary(sep_east_fit_2sd)
summary(sep_east_fit_3sd)

# unfortunately some of these are nans, so this line doesn't work all that well. 

pred_interval_1sd <- cbind(predict(sep_east_fit, interval="prediction", level = 0.68), sep_east_dfs$Org_H_md)
pred_interval_2sd <- cbind(predict(sep_east_fit, interval="prediction", level = 0.95), sep_east_dfs$Org_H_md)
pred_interval_3sd <- cbind(predict(sep_east_fit, interval="prediction", level = 0.997), sep_east_dfs$Org_H_md)

pred_interval_1sd <- pred_interval_1sd[ order(pred_interval_1sd[,4]),]
pred_interval_2sd <- pred_interval_2sd[ order(pred_interval_2sd[,4]),]
pred_interval_3sd <- pred_interval_3sd[ order(pred_interval_3sd[,4]),]

pred_interval_3sd

plot(Chng_H_md ~ Org_H_md, data=sep_dfs)
lines(pred_interval_1sd[,4], pred_interval_1sd[, "upr"], col="blue", lty=3)
lines(pred_interval_1sd[,4], pred_interval_1sd[, "lwr"], col="blue", lty=3)
lines(pred_interval_2sd[,4], pred_interval_2sd[, "upr"], col="blue", lty=3)
lines(pred_interval_2sd[,4], pred_interval_2sd[, "lwr"], col="blue", lty=3)
lines(pred_interval_3sd[,4], pred_interval_3sd[, "upr"], col="blue", lty=3)
lines(pred_interval_3sd[,4], pred_interval_3sd[, "lwr"], col="blue", lty=3)

sep_east_model_intercept <- coef(sep_east_fit)[1]
sep_east_model_slope <- coef(sep_east_fit)[2]

sep_east_dfs$Org_H_max_round=round(sep_east_dfs$Org_H_mx,digits=-1)
sep_east_dfs$Org_H_md_round=round(sep_east_dfs$Org_H_md,digits=-1)

sep_east_df_summary= sep_east_dfs@data %>% group_by(Site,Org_H_md_round) %>% 
  summarize(num_tot=n(),num_died=sum(Chng_H_md <= (Org_H_md*-0.061)-2.2, na.rm=TRUE),pct_died_per_year=(100*num_died/num_tot)^(1/5.314))

sep_east_df_summary$pct_died_per_year


sep_east_df_base = sep_east_dfs@data %>% group_by(Site,Org_H_md_round)

sep_east_df_base_2 = sep_east_df_base %>% summarize(num_tot=n(),
                            num_died=sum(Chng_H_md <= (Org_H_md*-0.061)-2.2, 
                            na.rm=TRUE),
                            pct_died_per_year=(100*num_died/num_tot)^(1/5.314))

sep_east_df_base_2

sep_east_dfs@data[2000,]

colSums(is.na(sep_east_dfs@data))
# x1 x2 x3 
#  2  1  0

# bootstrapping...

library(boot)

#define function to calculate fitted regression coefficients
mortality_function <- function(data, indices){
  d <- data[indices,] #allows boot to select sample
  mort <- d %>% group_by(Org_H_md_round) %>% 
    summarize(num_tot=n(),num_died=sum(Chng_H_md <= (Org_H_md*-0.061)-2.2),
              pct_died_per_year=(100*num_died/num_tot)^(1/5.314)) #fit regression model
  return(mort$pct_died_per_year[])
  #return coefficient estimates of model
}

mort <- sep_east_dfs@data %>% group_by(Org_H_md_round) %>% 
  summarize(num_tot=n(),num_died=sum(Chng_H_md <= (Org_H_md*-0.061)-2.2, na.rm=TRUE),
            pct_died_per_year=((100*num_died)/num_tot)^(1/5.314))

mort$pct_died_per_year[4]

mortality_function <- function(data, indices){
  d <- data[indices,] #allows boot to select sample
  mort <- d %>% group_by(Org_H_md_round) %>% 
    summarize(num_tot=n(),num_died=sum(Chng_H_md <= (Org_H_md*-0.061)-2.2, na.rm=TRUE),
              pct_died_per_year=((100*num_died)/num_tot)^(1/5.314)) #fit regression model
  return(mort$pct_died_per_year[7])
  #return coefficient estimates of model
}


mortality_function(sep_east_dfs@data)

reps <- boot(data=sep_east_dfs@data, statistic=mortality_function, R=1000)

reps


plot(reps)

boot.ci(boot.out=reps, type="bca")

## bootstrapping of sep east growth
grow <- sep_east_dfs@data %>% group_by(Org_H_md_round) %>% 
  summarize(num_tot=n(),growth=sum(Chng_H_md >= (Org_H_md*-0.061)-2.2, na.rm=TRUE)/(5.314*num_tot)) 

grow


growth_function <- function(data, indices){
  d <- data[indices,] #allows boot to select sample
  grow <- d %>% group_by(Org_H_md_round) %>% 
    summarize(num_tot=n(),growth=sum(Chng_H_md >= (Org_H_md*-0.061)-2.2, na.rm=TRUE)/(5.314*num_tot))      
  return(grow$growth[7])
  #return coefficient estimates of model
}


growth_function(sep_east_dfs@data)

reps <- boot(data=sep_east_dfs@data, statistic=growth_function, R=1000)

reps

plot(reps)

##plotting


sep_east_df_summary

ggplot(sep_east_df_summary,aes(Org_H_md_round, pct_died_per_year))+
  geom_point(color='red',aes(Org_H_md_round,pct_died_per_year))+
  ggtitle('Sepilok, annual mortality rate of Mask R-CNN trees')+
  theme_classic()+
  theme(plot.title=element_text(size=10,face='bold',margin=margin(10,10,10,0), hjust = 0.5))+
  labs(x='Binned tree heights / m', y='Annual tree mortality / %')+
  theme(legend.title = element_text(colour="black", size=12, face="bold"))+
  xlim(10,60)+
  ylim(0, 4)

# let's just do a quick carbon storage for the first LiDAR scan

### Now do a carbon calculation

sep_east_dfs$diameter = 2*sqrt((sep_east_dfs$area_sqm)/pi)

sep_east_dfs$agb = 0.136 * (sep_east_dfs$Org_H_mx * sep_east_dfs$diameter)^1.52

sep_east_dfs$carbon = 0.5 * sep_east_dfs$agb

sep_east_sum_carbon = sum(sep_east_dfs$carbon, na.rm=TRUE)

summary(sep_east_dfs)

ggplot()+
  geom_point(color='red', data=sep_east_dfs@data, aes(Org_H_md, carbon, size=area), alpha=0.2)+
  ylim(0, 20000)+ggtitle('Carbon stored in each tree predicted by Mask R-CNN')+
  theme_classic()+
  theme(plot.title=element_text(size=10,face='bold',margin=margin(10,10,10,0), hjust = 0.5))+
  labs(x='Tree height / m', y='Carbon / kg')+
  theme(legend.title = element_text(colour="black", size=8, face="bold"))+
  labs(size='Crown Area')+
  guides(size = guide_legend(reverse=TRUE))

#########################################################
# SEPILOK WEST
#########################################################

sep_west_dfs$area_sqm <- raster::area(sep_west_dfs) 

# let's filter out the trees smaller than 10 metres
sep_west_dfs = sep_west_dfs[sep_west_dfs$Org_H_md > 10,]

# try to filter out ones that are too large
sep_west_dfs = sep_west_dfs[sep_west_dfs$prmtr > 5,]
sep_west_dfs = sep_west_dfs[sep_west_dfs$area_sqm < 3000,]
sep_west_dfs = sep_west_dfs[sep_west_dfs$area_sqm > 5,]

ggplot(sep_west_dfs@data,aes(Org_H_md, Chng_H_md))+
  geom_point(color='red', aes(Org_H_md,Chng_H_md))+
  geom_smooth(method='rlm', aes(Org_H_md,Chng_H_md))+
  ggtitle('Sepilok, change in tree height predicted by Mask R-CNN')+
  theme_classic()+
  theme(plot.title=element_text(size=10,face='bold',margin=margin(10,10,10,0), hjust = 0.5))+
  labs(x='2014 tree height / m', y='Change in tree height / m')+
  theme(legend.title = element_text(colour="black", size=8, face="bold"))+
  labs(size='Crown Area')+
  guides(size = guide_legend(reverse=TRUE))+
  xlim(-10,80)+
  ylim(-60, 40)


sep_west_fit <- rlm(Chng_H_md ~ Org_H_md, data = sep_west_dfs@data)
summary(sep_west_fit)

sep_west_fit_1sd <- predict(sep_west_fit, interval="prediction", level = 0.68)
sep_west_fit_2sd <- predict(sep_west_fit, interval="prediction", level = 0.95)
sep_west_fit_3sd <- predict(sep_west_fit, interval="prediction", level = 0.997)

summary(sep_west_fit_1sd)
summary(sep_west_fit_2sd)
summary(sep_west_fit_3sd)

pred_interval_1sd <- cbind(predict(sep_west_fit, interval="prediction", level = 0.68), sep_west_dfs$Org_H_md)
pred_interval_2sd <- cbind(predict(sep_west_fit, interval="prediction", level = 0.95), sep_west_dfs$Org_H_md)
pred_interval_3sd <- cbind(predict(sep_west_fit, interval="prediction", level = 0.997), sep_west_dfs$Org_H_md)

pred_interval_1sd <- pred_interval_1sd[ order(pred_interval_1sd[,4]),]
pred_interval_2sd <- pred_interval_2sd[ order(pred_interval_2sd[,4]),]
pred_interval_3sd <- pred_interval_3sd[ order(pred_interval_3sd[,4]),]



plot(Chng_H_md ~ Org_H_md, data=sep_dfs)
lines(pred_interval_1sd[,4], pred_interval_1sd[, "upr"], col="blue", lty=3)
lines(pred_interval_1sd[,4], pred_interval_1sd[, "lwr"], col="blue", lty=3)
lines(pred_interval_2sd[,4], pred_interval_2sd[, "upr"], col="blue", lty=3)
lines(pred_interval_2sd[,4], pred_interval_2sd[, "lwr"], col="blue", lty=3)
lines(pred_interval_3sd[,4], pred_interval_3sd[, "upr"], col="blue", lty=3)
lines(pred_interval_3sd[,4], pred_interval_3sd[, "lwr"], col="blue", lty=3)

sep_west_model_intercept <- coef(sep_west_fit)[1]
sep_west_model_slope <- coef(sep_west_fit)[2]

sep_west_dfs$Org_H_max_round=round(sep_west_dfs$Org_H_mx,digits=-1)
sep_west_dfs$Org_H_md_round=round(sep_west_dfs$Org_H_md,digits=-1)

sep_west_df_summary= sep_west_dfs@data %>% group_by(Site,Org_H_md_round) %>% 
  summarize(num_tot=n(),num_died=sum(Chng_H_md <= (Org_H_md*-0.0728)-2.2, na.rm=TRUE),pct_died_per_year=(100*num_died/num_tot)^(1/5.314))

## bootstrapping of sep west 

mort <- sep_west_dfs@data %>% group_by(Org_H_md_round) %>% 
  summarize(num_tot=n(),num_died=sum(Chng_H_md <= (Org_H_md*-0.0728)-2.2, na.rm=TRUE),
            pct_died_per_year=((100*num_died)/num_tot)^(1/5.314))

mort

mortality_function <- function(data, indices){
  d <- data[indices,] #allows boot to select sample
  mort <- d %>% group_by(Org_H_md_round) %>% 
    summarize(num_tot=n(),num_died=sum(Chng_H_md <= (Org_H_md*-0.0728)-2.2, na.rm=TRUE),
              pct_died_per_year=((100*num_died)/num_tot)^(1/5.314)) #fit regression model
  return(mort$pct_died_per_year[2])
  #return coefficient estimates of model
}


mortality_function(sep_west_dfs@data)

reps <- boot(data=sep_west_dfs@data, statistic=mortality_function, R=1000)

reps


plot(reps)

## bootstrapping of sep west growth
grow <- sep_west_dfs@data %>% group_by(Org_H_md_round) %>% 
  summarize(num_tot=n(),growth=sum(Chng_H_md >= (Org_H_md*-0.0728)-2.2, na.rm=TRUE)/(5.314*num_tot)) 

grow

growth_function <- function(data, indices){
  d <- data[indices,] #allows boot to select sample
  grow <- d %>% group_by(Org_H_md_round) %>% 
    summarize(num_tot=n(),growth=sum(Chng_H_md >= (Org_H_md*-0.0728)-2.2, na.rm=TRUE)/(5.314*num_tot))      
  return(grow$growth[6])
  #return coefficient estimates of model
}


growth_function(sep_west_dfs@data)

reps <- boot(data=sep_west_dfs@data, statistic=growth_function, R=1000)

reps

plot(reps)

##plotting



ggplot(sep_west_df_summary,aes(Org_H_md_round, pct_died_per_year))+
  geom_point(color='red',aes(Org_H_md_round,pct_died_per_year))+
  ggtitle('Sepilok, annual mortality rate of Mask R-CNN trees')+
  theme_classic()+
  theme(plot.title=element_text(size=10,face='bold',margin=margin(10,10,10,0), hjust = 0.5))+
  labs(x='Binned tree heights / m', y='Annual tree mortality / %')+
  theme(legend.title = element_text(colour="black", size=12, face="bold"))+
  xlim(10,60)+
  ylim(0, 4)

# let's just do a quick carbon storage for the first LiDAR scan

### Now do a carbon calculation


sep_west_dfs$diameter = 2*sqrt((sep_west_dfs$area_sqm)/pi)

sep_west_dfs$agb = 0.136 * (sep_west_dfs$Org_H_mx * sep_west_dfs$diameter)^1.52

sep_west_dfs$carbon = 0.5 * sep_west_dfs$agb

sep_west_sum_carbon = sum(sep_west_dfs$carbon, na.rm=TRUE)

summary(sep_west_dfs)

ggplot()+
  geom_point(color='red', data=sep_west_dfs@data, aes(Org_H_md, carbon, size=area), alpha=0.2)+
  ylim(0, 20000)+ggtitle('Carbon stored in each tree predicted by Mask R-CNN')+
  theme_classic()+
  theme(plot.title=element_text(size=10,face='bold',margin=margin(10,10,10,0), hjust = 0.5))+
  labs(x='Tree height / m', y='Carbon / kg')+
  theme(legend.title = element_text(colour="black", size=8, face="bold"))+
  labs(size='Crown Area')+
  guides(size = guide_legend(reverse=TRUE))




# Danum
dan_dfs = shapefile("C:/Users/sebhi/ai4er/mres_project/work/final_dfs/dan_dfs.shp")


# Let's have a look at Sepilok first
# Read in the rasters...

Dan_2014 = raster(paste0("C:/Users/sebhi/ai4er/mres_project/work/data/danum/danum_lidar/Danum_2014_CHM_g10_sub0.2_1m.tif"))
Dan_2020 = raster(paste0("C:/Users/sebhi/ai4er/mres_project/work/data/danum/danum_lidar/Danum_2020_CHM_g10_sub0.2_1m.tif"))
Dan_diff = raster(paste0("C:/Users/sebhi/ai4er/mres_project/work/data/danum/danum_lidar/Danum_dDSM_random_1m.tif"))

plot(Dan_diff)
plot(dan_dfs, add=TRUE)


# let's have a look at a summary of our crowns
summary(dan_dfs)

# let's filter out all trees smaller than 10m tall

dan_dfs$area_sqm <- raster::area(dan_dfs) 

dan_dfs = dan_dfs[dan_dfs$Org_H_md > 10,]

ggplot(dan_dfs@data,aes(Org_H_md, Chng_H_md))+
  geom_point(color='red', aes(Org_H_md,Chng_H_md))+
  geom_smooth(method='rlm', data=dan_dfs@data, aes(Org_H_md, Chng_H_md), color='darkgreen', alpha=1)+
  ggtitle('Danum, change in tree height predicted by Mask R-CNN')+
  theme_classic()+
  theme(plot.title=element_text(size=10,face='bold',margin=margin(10,10,10,0), hjust = 0.5))+
  labs(x='2014 tree height / m', y='Change in tree height / m')+
  theme(legend.title = element_text(colour="black", size=8, face="bold"))+
  labs(size='Crown Area')+
  guides(size = guide_legend(reverse=TRUE))+
  xlim(-10,80)+
  ylim(-60, 40)


dan_fit <- rlm(Chng_H_md ~ Org_H_md, data = dan_dfs@data)
summary(dan_fit)

dan_fit_1sd <- predict(dan_fit, interval="prediction", level = 0.68)
dan_fit_2sd <- predict(dan_fit, interval="prediction", level = 0.95)
dan_fit_3sd <- predict(dan_fit, interval="prediction", level = 0.997)

summary(dan_fit_1sd)
summary(dan_fit_2sd)
summary(dan_fit_3sd)


### NEED TO ADD THE LAST CLAUSE TO THE SEPILOK ONES ABOVE
dan_pred_interval_1sd <- cbind(predict(dan_fit, interval="prediction", level = 0.68), dan_dfs$Org_H_md)
dan_pred_interval_2sd <- cbind(predict(dan_fit, interval="prediction", level = 0.95), dan_dfs$Org_H_md)
dan_pred_interval_3sd <- cbind(predict(dan_fit, interval="prediction", level = 0.997), dan_dfs$Org_H_md)

dan_pred_interval_1sd <- dan_pred_interval_1sd[order(dan_pred_interval_1sd[,4]),]
dan_pred_interval_2sd <- dan_pred_interval_2sd[order(dan_pred_interval_2sd[,4]),]
dan_pred_interval_3sd <- dan_pred_interval_3sd[order(dan_pred_interval_3sd[,4]),]

plot(Chng_H_md ~ Org_H_md, data=dan_dfs)
lines(dan_pred_interval_1sd[,4], dan_pred_interval_1sd[, "upr"], col="blue", lty=3)
lines(dan_pred_interval_1sd[,4], dan_pred_interval_1sd[, "lwr"], col="blue", lty=3)
lines(dan_pred_interval_2sd[,4], dan_pred_interval_2sd[, "upr"], col="blue", lty=3)
lines(dan_pred_interval_2sd[,4], dan_pred_interval_2sd[, "lwr"], col="blue", lty=3)
lines(dan_pred_interval_3sd[,4], dan_pred_interval_3sd[, "upr"], col="blue", lty=3)
lines(dan_pred_interval_3sd[,4], dan_pred_interval_3sd[, "lwr"], col="blue", lty=3)

dan_model_intercept <- coef(dan_fit)[1]
dan_model_slope <- coef(dan_fit)[2]

dan_dfs$Org_H_max_round=round(dan_dfs$Org_H_mx,digits=-1)
dan_dfs$Org_H_md_round=round(dan_dfs$Org_H_md,digits=-1)

dan_df_summary= dan_dfs@data %>% group_by(Site,Org_H_md_round) %>% 
  summarize(num_tot=n(),num_died=sum(Chng_H_md <= (Org_H_md*-0.11)-11.1, na.rm=TRUE),pct_died_per_year=(100*num_died/num_tot)^(1/5.361))

## bootstrapping of danum

mort <- dan_dfs@data %>% group_by(Org_H_md_round) %>% 
  summarize(num_tot=n(),num_died=sum(Chng_H_md <= (Org_H_md*-0.11)-11.1, na.rm=TRUE),
            pct_died_per_year=((100*num_died)/num_tot)^(1/5.361))

mort

mortality_function <- function(data, indices){
  d <- data[indices,] #allows boot to select sample
  mort <- d %>% group_by(Org_H_md_round) %>% 
    summarize(num_tot=n(),num_died=sum(Chng_H_md <= (Org_H_md*-0.11)-11.1, na.rm=TRUE),
              pct_died_per_year=((100*num_died)/num_tot)^(1/5.361)) #fit regression model
  return(mort$pct_died_per_year[7])
  #return coefficient estimates of model
}


mortality_function(dan_dfs@data)

reps <- boot(data=dan_dfs@data, statistic=mortality_function, R=1000)

reps

plot(reps)

## bootstrapping of danum growth 
grow <- dan_dfs@data %>% group_by(Org_H_md_round) %>% 
  summarize(num_tot=n(),growth=sum(Chng_H_md >= (Org_H_md*-0.11)-11.1, na.rm=TRUE)/(5.361*num_tot)) 

grow

growth_function <- function(data, indices){
  d <- data[indices,] #allows boot to select sample
  grow <- d %>% group_by(Org_H_md_round) %>% 
    summarize(num_tot=n(),growth=sum(Chng_H_md >= (Org_H_md*-0.11)-11.1, na.rm=TRUE)/(5.361*num_tot))      
  return(grow$growth[8])
  #return coefficient estimates of model
}


growth_function(dan_dfs@data)

reps <- boot(data=dan_dfs@data, statistic=growth_function, R=1000)

reps

plot(reps)

##plotting

ggplot(dan_df_summary,aes(Org_H_md_round, pct_died_per_year))+
  geom_point(color='red',aes(Org_H_md_round,pct_died_per_year))+
  ggtitle('Danum, annual mortality rate of Mask R-CNN trees')+
  theme_classic()+
  theme(plot.title=element_text(size=10,face='bold',margin=margin(10,10,10,0), hjust = 0.5))+
  labs(x='Binned tree heights / m', y='Annual tree mortality / %')+
  theme(legend.title = element_text(colour="black", size=12, face="bold"))+
  xlim(15,60)+
  ylim(0, 4)


### Now do a carbon calculation


dan_dfs$diameter = 2*sqrt((dan_dfs$area_sqm)/pi)

dan_dfs$agb = 0.136 * ((dan_dfs$Org_H_mx * dan_dfs$diameter))^1.52

dan_dfs$carbon = 0.5 * dan_dfs$agb

dan_sum_carbon = sum(dan_dfs$carbon, na.rm=TRUE)


ggplot()+
  geom_point(color='red', data=dan_dfs@data, aes(Org_H_md, carbon, size=area), alpha=0.2)+
  ylim(0, 20000)+ggtitle('Carbon stored in each tree predicted by Mask R-CNN')+
  theme_classic()+
  theme(plot.title=element_text(size=10,face='bold',margin=margin(10,10,10,0), hjust = 0.5))+
  labs(x='Tree height / m', y='Carbon / kg')+
  theme(legend.title = element_text(colour="black", size=8, face="bold"))+
  labs(size='Crown Area')+
  guides(size = guide_legend(reverse=TRUE))



# PARACOU

par_dfs = shapefile("C:/Users/sebhi/ai4er/mres_project/work/final_dfs/par_dfs.shp")

# Let's have a look at Sepilok first
# Read in the rasters...

Par_2016 = raster(paste0("C:/Users/sebhi/ai4er/mres_project/work/data/paracou/lidar/Par_2016_CHM_g10_sub0.2_1m.tif"))
Par_2019 = raster(paste0("C:/Users/sebhi/ai4er/mres_project/work/data/paracou/lidar/Par_2019_CHM_g10_sub0.2_1m.tif"))
Par_diff = Par_2019 - Par_2016

### Plot the crowns and the raster

plot(Par_diff)
plot(par_dfs, add=TRUE)

# let's have a look at a summary of our crowns
summary(par_dfs)

par_dfs$area_sqm <- raster::area(par_dfs) 

par_dfs = par_dfs[!is.na(par_dfs$Org_H_md), ]
par_dfs = par_dfs[par_dfs$Org_H_md > 10,]

plot(Par_diff)
plot(par_dfs, add=TRUE)


# put horizontal line on this ggplot

ggplot(par_dfs@data,aes(Org_H_md, Chng_H_md))+
  geom_point(color='red', aes(Org_H_md,Chng_H_md), alpha = 0.3)+
  ggtitle('Paracou, change in tree height predicted by Mask R-CNN')+
  theme_classic()+
  theme(plot.title=element_text(size=10,face='bold',margin=margin(10,10,10,0), hjust = 0.5))+
  labs(x='2016 tree height / m', y='Change in tree height / m')+
  theme(legend.title = element_text(colour="black", size=8, face="bold"))+
  labs(size='Crown Area')+
  guides(size = guide_legend(reverse=TRUE))+
  xlim(-10,80)+
  ylim(-60, 20)

### Fresher plot

ggplot(par_dfs@data, aes(Org_H_md, Chng_H_md))+
  geom_pointdensity(aes(Org_H_md, Chng_H_md), alpha = 0.8, shape='.')+
  geom_hline(yintercept=0, color="orange", linetype=2)+
  geom_abline(intercept = 0.32, slope = -0.028, color='navy', alpha = 0.8, linetype='dashed')+
  geom_abline(intercept = 1.8, slope = -0.028, color='navy', alpha = 0.8, linetype='dashed')+
  geom_smooth(method='rlm', aes(Org_H_md, Chng_H_md), size = 0.5, color='red')+
  theme_light()+
  scale_color_manual(values=c(brewer.pal(8,"Dark2")[c(8,1)],brewer.pal(10,"Paired")[c(9,10)]))+
  labs(x='2014 tree height (m)', y='Change in tree height (m)')+
  #theme(legend.title = element_text(colour="black", size=8, face="bold"))+
  #labs(size='Crown Area')+
  xlim(0,65)+
  ylim(-40, 10)

ggsave("C:/Users/sebhi/ai4er/paracou_growth_by_height.png")


#### Altered plot with highlighted mortalities

# filter dataframe to get data to be highlighted
highlight_df <- par_dfs@data %>% 
  filter(Chng_H_md <= (Org_H_md*-0.0279)+0.315)

highlight_df

ggplot(par_dfs@data, aes(Org_H_md, Chng_H_md))+
  geom_point(aes(Org_H_md, Chng_H_md), color='darkgrey', alpha = 0.8, shape='.')+
  geom_point(data=highlight_df, aes(x=Org_H_md,y=Chng_H_md), 
             color='red', shape='.')+
  geom_hline(yintercept=0, color="orange", linetype=3)+
  geom_abline(intercept = 0.32, slope = -0.028, color='navy', alpha = 0.8, linetype='dashed')+
  geom_abline(intercept = 1.8, slope = -0.028, color='navy', alpha = 0.8, linetype='dashed')+
  geom_smooth(method='rlm', aes(Org_H_md, Chng_H_md), size = 0.5, color='red')+
  theme_light()+
  scale_color_manual(values=c(brewer.pal(8,"Dark2")[c(8,1)],brewer.pal(10,"Paired")[c(9,10)]))+
  labs(x='Tree height, Sept 2016 (m)', y='Change in tree height, Nov 2019 (m)')+
  annotate("text", x=63, y=7, label= "(A)", size=10)+
  #theme(legend.title = element_text(colour="black", size=8, face="bold"))+
  #labs(size='Crown Area')+
  xlim(0,65)+
  ylim(-40, 10)

ggsave("C:/Users/sebhi/ai4er/plots_for_manuscript/paracou_growth_by_height_colour_deaths_dated_axes.png", width=5, height=4)


par_fit <- rlm(Chng_H_md ~ Org_H_md, data = par_dfs@data)
summary(par_fit)

par_fit_1sd <- predict(par_fit, interval="prediction", level = 0.68)
par_fit_2sd <- predict(par_fit, interval="prediction", level = 0.95)
par_fit_3sd <- predict(par_fit, interval="prediction", level = 0.997)

summary(par_fit_1sd)
summary(par_fit_2sd)
summary(par_fit_3sd)

par_pred_interval_1sd <- cbind(predict(par_fit, interval="prediction", level = 0.68), par_dfs$Org_H_md)
par_pred_interval_2sd <- cbind(predict(par_fit, interval="prediction", level = 0.95), par_dfs$Org_H_md)
par_pred_interval_3sd <- cbind(predict(par_fit, interval="prediction", level = 0.997), par_dfs$Org_H_md)

par_pred_interval_1sd <- par_pred_interval_1sd[order(par_pred_interval_1sd[,4]),]
par_pred_interval_2sd <- par_pred_interval_2sd[order(par_pred_interval_2sd[,4]),]
par_pred_interval_3sd <- par_pred_interval_3sd[order(par_pred_interval_3sd[,4]),]

plot(Chng_H_md ~ Org_H_md, data=par_dfs)
lines(par_pred_interval_1sd[,4], par_pred_interval_1sd[, "upr"], col="blue", lty=3)
lines(par_pred_interval_1sd[,4], par_pred_interval_1sd[, "lwr"], col="blue", lty=3)
lines(par_pred_interval_2sd[,4], par_pred_interval_2sd[, "upr"], col="blue", lty=3)
lines(par_pred_interval_2sd[,4], par_pred_interval_2sd[, "lwr"], col="blue", lty=3)
lines(par_pred_interval_3sd[,4], par_pred_interval_3sd[, "upr"], col="blue", lty=3)
lines(par_pred_interval_3sd[,4], par_pred_interval_3sd[, "lwr"], col="blue", lty=3)

par_model_intercept <- coef(par_fit)[1]
par_model_slope <- coef(par_fit)[2]

ggplot(data = par_dfs@data, aes(Org_H_md, Chng_H_md, color=Site)) +
  geom_point() +
  geom_abline(intercept = par_model_intercept, slope = par_model_slope, color = 'black') + xlim(0,80)

par_dfs$Org_H_max_round=round(par_dfs$Org_H_mx,digits=-1)
par_dfs$Org_H_md_round=round(par_dfs$Org_H_md,digits=-1)

par_df_summary= par_dfs@data %>% group_by(Site,Org_H_md_round) %>% 
  summarize(num_tot=n(),num_died=sum(Chng_H_md <= (Org_H_md*-0.0279)+0.2, na.rm=TRUE),pct_died_per_year=(100*num_died/num_tot)^(1/3.154))

## bootstrapping of paracou mortality

mort <- par_dfs@data %>% group_by(Org_H_md_round) %>% 
  summarize(num_tot=n(),num_died=sum(Chng_H_md <= (Org_H_md*-0.0279)+0.2, na.rm=TRUE),
            pct_died_per_year=((100*num_died)/num_tot)^(1/3.154))

mort

mortality_function <- function(data, indices){
  d <- data[indices,] #allows boot to select sample
  mort <- d %>% group_by(Org_H_md_round) %>% 
    summarize(num_tot=n(),num_died=sum(Chng_H_md <= (Org_H_md*-0.0279)+0.2, na.rm=TRUE),
              pct_died_per_year=((100*num_died)/num_tot)^(1/3.154)) #fit regression model
  return(mort$pct_died_per_year[5])
  #return coefficient estimates of model
}


mortality_function(par_dfs@data)

reps <- boot(data=par_dfs@data, statistic=mortality_function, R=1000)

reps

plot(reps)

## bootstrapping of paracou growth

grow <- par_dfs@data %>% group_by(Org_H_md_round) %>% 
  summarize(num_tot=n(),growth=sum(Chng_H_md >= (Org_H_md*-0.0279)+0.2, na.rm=TRUE)/(3.154*num_tot))

grow

growth_function <- function(data, indices){
  d <- data[indices,] #allows boot to select sample
  grow <- d %>% group_by(Org_H_md_round) %>% 
      summarize(num_tot=n(),growth=sum(Chng_H_md >= (Org_H_md*-0.0279)+0.2, na.rm=TRUE)/(3.154*num_tot))      
  return(grow$growth[6])
  #return coefficient estimates of model
}


growth_function(par_dfs@data)

reps <- boot(data=par_dfs@data, statistic=growth_function, R=1000)

reps

plot(reps)

# back to plotting 

ggplot(par_df_summary,aes(Org_H_md_round, pct_died_per_year))+
  geom_point(color='red',aes(Org_H_md_round,pct_died_per_year))+
  ggtitle('Paracou, annual mortality rate of Mask R-CNN trees')+
  theme_classic()+
  theme(plot.title=element_text(size=10,face='bold',margin=margin(10,10,10,0), hjust = 0.5))+
  labs(x='Binned tree heights / m', y='Annual tree mortality / %')+
  theme(legend.title = element_text(colour="black", size=12, face="bold"))+
  xlim(10,60)+
  ylim(0, 6)

### Now do a carbon calculation


par_dfs$diameter = 2*sqrt((par_dfs$area_sqm)/pi)

par_dfs$agb = 0.136 * (par_dfs$Org_H_mx * par_dfs$diameter)^1.52

par_dfs$carbon = 0.5 * par_dfs$agb

par_sum_carbon = sum(par_dfs$carbon, na.rm=TRUE)


ggplot()+
  geom_point(color='red', data=par_dfs@data, aes(Org_H_md, carbon, size=area), alpha=0.1)+
  ylim(0, 20000)+ggtitle('Carbon stored in each tree predicted by Mask R-CNN')+
  theme_classic()+
  theme(plot.title=element_text(size=10,face='bold',margin=margin(10,10,10,0), hjust = 0.5))+
  labs(x='Tree height / m', y='Carbon / kg')+
  theme(legend.title = element_text(colour="black", size=8, face="bold"))+
  labs(size='Crown Area')+
  guides(size = guide_legend(reverse=TRUE))


save(par_df_summary, file = "C:/Users/sebhi/ai4er/mres_project/par_df_summary.Rda")     
save(dan_df_summary, file = "C:/Users/sebhi/ai4er/mres_project/dan_df_summary.Rda")     
save(sep_west_df_summary, file = "C:/Users/sebhi/ai4er/mres_project/sep_west_df_summary.Rda")     
save(sep_east_df_summary, file = "C:/Users/sebhi/ai4er/mres_project/sep_east_df_summary.Rda")     

######################################
### Joint plots
######################################

### first growth against original tree height

ggplot()+
  geom_point(data=par_dfs@data, aes(Org_H_md,Chng_H_md, color=Site), alpha=0.4)+
  geom_point(data=sep_east_dfs@data, aes(Org_H_md,Chng_H_md, color=Site), alpha=0.5)+
  geom_point(data=sep_west_dfs@data, aes(Org_H_md,Chng_H_md, color=Site), alpha=0.5)+
  geom_point(data=dan_dfs@data, aes(Org_H_md,Chng_H_md, color=Site), alpha=0.2)+
  geom_smooth(method='rlm', data=par_dfs@data, aes(Org_H_md,Chng_H_md), color='darkgreen', alpha=1)+
  geom_smooth(method='rlm', data=sep_east_dfs@data, aes(Org_H_md,Chng_H_md), color='navy', alpha=0.6)+
  geom_smooth(method='rlm', data=sep_west_dfs@data, aes(Org_H_md,Chng_H_md), color='purple', alpha=0.6)+
  geom_smooth(method='rlm', data=dan_dfs@data, aes(Org_H_md,Chng_H_md), color='darkred', alpha=1)+
  geom_hline(yintercept=0, linetype='dashed', size=0.5)+
  ggtitle('Change in tree height per site predicted by Mask R-CNN')+
  theme_classic()+
  theme(plot.title=element_text(size=10,face='bold',margin=margin(10,10,10,0), hjust = 0.5))+
  labs(x='Original tree height / m', y='Change in tree height / m')+
  theme(legend.title = element_text(colour="black", size=8, face="bold"))+
  labs(size='Crown Area')+
  guides(size = guide_legend(reverse=TRUE))+
  xlim(-10,80)+
  ylim(-60, 20)

ggsave("C:/Users/sebhi/ai4er/tree_growth_small_four_sites.png")



## then joint mortality plot

summary(par_df_summary)

ggplot()+
  geom_point(data = par_df_summary, aes(Org_H_md_round, pct_died_per_year, color=Site))+
  geom_point(data = sep_east_df_summary, aes(Org_H_md_round, pct_died_per_year, color=Site))+
  geom_point(data = sep_west_df_summary, aes(Org_H_md_round, pct_died_per_year, color=Site))+
  geom_point(data = dan_df_summary, aes(Org_H_md_round, pct_died_per_year, color=Site))+
  geom_line(data = par_df_summary, aes(Org_H_md_round, pct_died_per_year, color=Site))+
  geom_line(data = sep_east_df_summary, aes(Org_H_md_round, pct_died_per_year, color=Site))+
  geom_line(data = sep_west_df_summary, aes(Org_H_md_round, pct_died_per_year, color=Site))+
  geom_line(data = dan_df_summary, aes(Org_H_md_round, pct_died_per_year, color=Site))+
  geom_errorbar(data = par_df_summary, aes(x=Org_H_md_round, ymin=pct_died_per_year-0.1, ymax=pct_died_per_year+0.1, color=Site), width=.2, position=position_dodge(0.05))+
  geom_errorbar(data = sep_east_df_summary, aes(x=Org_H_md_round, ymin=pct_died_per_year-0.2, ymax=pct_died_per_year+0.2, color=Site), width=.2, position=position_dodge(0.05))+
  geom_errorbar(data = sep_west_df_summary, aes(x=Org_H_md_round, ymin=pct_died_per_year-0.2, ymax=pct_died_per_year+0.2, color=Site), width=.2, position=position_dodge(0.05))+
  geom_errorbar(data = dan_df_summary, aes(x=Org_H_md_round, ymin=pct_died_per_year-0.12, ymax=pct_died_per_year+0.12, color=Site), width=.2, position=position_dodge(0.05))+
  
  ggtitle('Annual mortality rate of Mask R-CNN trees')+
  theme_classic()+
  
  theme(plot.title=element_text(size=10,face='bold',margin=margin(10,10,10,0), hjust = 0.5))+
  labs(x='Binned tree heights / m', y='Annual tree mortality / %')+
  theme(legend.title = element_text(colour="black", size=12, face="bold"))+
  xlim(15,60)+
  ylim(0, 4)

ggsave("C:/Users/sebhi/ai4er/mortality_small_four_sites.png")

# tree count raw

ggplot()+  
  geom_col(data = par_df_summary, aes(Org_H_md_round, num_tot, color=Site))+
  geom_col(data = sep_east_df_summary, aes(Org_H_md_round, num_tot, color=Site))+
  geom_col(data = sep_west_df_summary, aes(Org_H_md_round, num_tot, color=Site))+
  geom_col(data = dan_df_summary, aes(Org_H_md_round, num_tot, color=Site))+
  ggtitle('Tree count in each height bin')+
  theme_classic()+
  theme(plot.title=element_text(size=10,face='bold',margin=margin(10,10,10,0), hjust = 0.5))+
  labs(x='Binned tree heights / m', y='Tree count')+
  theme(legend.title = element_text(colour="black", size=12, face="bold"))

ggsave("C:/Users/sebhi/ai4er/tree_count_small.png")





# tree count per height bin

ggplot()+  
  geom_col(data = par_dfs, aes(Org_H_md_round, carbon, color=Site))+
  geom_col(data = sep_east_df_summary, aes(Org_H_md_round, num_tot, color=Site))+
  geom_col(data = sep_west_df_summary, aes(Org_H_md_round, num_tot, color=Site))+
  geom_col(data = dan_df_summary, aes(Org_H_md_round, num_tot, color=Site))+
  ggtitle('Tree count in each height bin')+
  theme_classic()+
  theme(plot.title=element_text(size=10,face='bold',margin=margin(10,10,10,0), hjust = 0.5))+
  labs(x='Binned tree heights / m', y='Tree count')+
  theme(legend.title = element_text(colour="black", size=12, face="bold"))

ggsave("C:/Users/sebhi/ai4er/tree_count_small.png")

### CARBON PER HEIGHT BIN

par_dfs = par_dfs[!is.na(par_dfs$carbon), ]
dan_dfs = dan_dfs[!is.na(dan_dfs$carbon), ]
sep_east_dfs = sep_east_dfs[!is.na(sep_east_dfs$carbon), ]
sep_west_dfs = sep_west_dfs[!is.na(sep_west_dfs$carbon), ]



par_df_carbon_summary= par_dfs@data %>% group_by(Site,Org_H_md_round) %>% 
  summarize(carbon=sum(carbon)/par_sum_carbon, num_tot=n(),num_died=sum(Chng_H_md <= (Org_H_md*-0.0279)-0.2, na.rm=TRUE),pct_died_per_year=(100*num_died/num_tot)^(1/3))

par_df_carbon_summary

dan_dfs$Org_H_md_round=round(dan_dfs$Org_H_md,digits=-1)

dan_df_carbon_summary= dan_dfs@data %>% group_by(Site,Org_H_md_round) %>% 
  summarize(carbon=sum(carbon)/dan_sum_carbon, num_tot=n(),num_died=sum(Chng_H_md <= (Org_H_md*-0.0279)-0.2, na.rm=TRUE),pct_died_per_year=(100*num_died/num_tot)^(1/3))

sep_east_dfs$Org_H_md_round=round(sep_east_dfs$Org_H_md,digits=-1)

sep_east_df_carbon_summary= sep_east_dfs@data %>% group_by(Site,Org_H_md_round) %>% 
  summarize(carbon=sum(carbon)/sep_east_sum_carbon, num_tot=n(),num_died=sum(Chng_H_md <= (Org_H_md*-0.0279)-0.2, na.rm=TRUE),pct_died_per_year=(100*num_died/num_tot)^(1/3))

sep_west_dfs$Org_H_md_round=round(sep_west_dfs$Org_H_md,digits=-1)

sep_west_df_carbon_summary= sep_west_dfs@data %>% group_by(Site,Org_H_md_round) %>% 
  summarize(carbon=sum(carbon)/sep_west_sum_carbon, num_tot=n(),num_died=sum(Chng_H_md <= (Org_H_md*-0.0279)-0.2, na.rm=TRUE),pct_died_per_year=(100*num_died/num_tot)^(1/3))




ggplot()+  
  geom_line(data = par_df_carbon_summary, aes(x=Org_H_md_round, y=carbon, color=Site))+
  geom_line(data = sep_east_df_carbon_summary, aes(x=Org_H_md_round, y=carbon, color=Site))+
  geom_line(data = sep_west_df_carbon_summary, aes(x=Org_H_md_round, y=carbon, color=Site))+
  geom_line(data = dan_df_carbon_summary, aes(x=Org_H_md_round, y=carbon, color=Site))+
  #geom_smooth(data = par_dfs@data, aes(x=Org_H_md, y=carbon, color=Site))+
  #geom_smooth(data = sep_east_dfs@data, aes(x=Org_H_md, y=carbon, color=Site))+
  #geom_smooth(data = sep_west_dfs@data, aes(x=Org_H_md, y=carbon, color=Site))+
  #geom_smooth(data = dan_dfs@data, aes(x=Org_H_md, y=carbon, color=Site))+
  ggtitle('Distribution of carbon in height bins')+
  theme_classic()+
  theme(plot.title=element_text(size=10,face='bold',margin=margin(10,10,10,0), hjust = 0.5))+
  labs(x='Tree height bins / m', y='Carbon / kg')+
  theme(legend.title = element_text(colour="black", size=12, face="bold"))

ggsave("C:/Users/sebhi/ai4er/carbon_per_height_bin_small.png")


#### TREE GROWTH PER HEIGHT BIN

##### Need to exclude trees below a certain threshold

par_df_growth_summary= par_dfs@data %>% group_by(Site,Org_H_md_round) %>% 
  summarize(num_tot=n(),growth=sum(Chng_H_md >= (Org_H_md*-0.0279)+0.2, na.rm=TRUE)/(3.154*num_tot))

dan_dfs$Org_H_md_round=round(dan_dfs$Org_H_md,digits=-1)

dan_df_growth_summary= dan_dfs@data %>% group_by(Site,Org_H_md_round) %>% 
  summarize(num_tot=n(),growth=sum(Chng_H_md >= (Org_H_md*-0.111)-11.1, na.rm=TRUE)/(5.361*num_tot))

sep_east_dfs$Org_H_md_round=round(sep_east_dfs$Org_H_md,digits=-1)

sep_east_df_growth_summary= sep_east_dfs@data %>% group_by(Site,Org_H_md_round) %>% 
  summarize(num_tot=n(),growth=sum(Chng_H_md >= (Org_H_md*-0.061)-2.2, na.rm=TRUE)/(5.314*num_tot))

sep_west_dfs$Org_H_md_round=round(sep_west_dfs$Org_H_md,digits=-1)

sep_west_df_growth_summary= sep_west_dfs@data %>% group_by(Site,Org_H_md_round) %>% 
  summarize(num_tot=n(),growth=sum(Chng_H_md >= (Org_H_md*-0.0728)-8.9, na.rm=TRUE)/(5.314*num_tot))

save(par_df_growth_summary, file = "C:/Users/sebhi/ai4er/mres_project/par_df_growth_summary.Rda")     
save(dan_df_growth_summary, file = "C:/Users/sebhi/ai4er/mres_project/dan_df_growth_summary.Rda")     
save(sep_east_df_growth_summary, file = "C:/Users/sebhi/ai4er/mres_project/sep_west_df_growth_summary.Rda")     
save(sep_west_df_growth_summary, file = "C:/Users/sebhi/ai4er/mres_project/sep_east_df_growth_summary.Rda")     





ggplot()+
  geom_point(data = par_df_growth_summary, aes(x=Org_H_md_round, y=growth, color=Site))+
  geom_point(data = sep_east_df_growth_summary, aes(x=Org_H_md_round, y=growth, color=Site))+
  geom_point(data = sep_west_df_growth_summary, aes(x=Org_H_md_round, y=growth, color=Site))+
  geom_point(data = dan_df_growth_summary, aes(x=Org_H_md_round, y=growth, color=Site))+
  geom_line(data = par_df_growth_summary, aes(x=Org_H_md_round, y=growth, color=Site))+
  geom_line(data = sep_east_df_growth_summary, aes(x=Org_H_md_round, y=growth, color=Site))+
  geom_line(data = sep_west_df_growth_summary, aes(x=Org_H_md_round, y=growth, color=Site))+
  geom_line(data = dan_df_growth_summary, aes(x=Org_H_md_round, y=growth, color=Site))+
  #geom_smooth(data = par_dfs@data, aes(x=Org_H_md, y=carbon, color=Site))+
  #geom_smooth(data = sep_east_dfs@data, aes(x=Org_H_md, y=carbon, color=Site))+
  #geom_smooth(data = sep_west_dfs@data, aes(x=Org_H_md, y=carbon, color=Site))+
  #geom_smooth(data = dan_dfs@data, aes(x=Org_H_md, y=carbon, color=Site))+
  xlim(15, 70)+
  ggtitle('Growth rate in height bins')+
  theme_classic()+
  theme(plot.title=element_text(size=10,face='bold',margin=margin(10,10,10,0), hjust = 0.5))+
  labs(x='Tree height bins / m', y='Growth / m yr-1')+
  theme(legend.title = element_text(colour="black", size=12, face="bold"))

ggsave("C:/Users/sebhi/ai4er/growth_per_height_bin_small.png")


### TREE HEIGHT DENSITY PLOT

ggplot()+  
  geom_density(data = par_dfs@data, aes(x=Org_H_md, color=Site))+
  geom_density(data = sep_east_dfs@data, aes(x=Org_H_md, color=Site))+
  geom_density(data = sep_west_dfs@data, aes(x=Org_H_md, color=Site))+
  geom_density(data = dan_dfs@data, aes(x=Org_H_md, color=Site))+
  ggtitle('Distribution of tree heights')+
  theme_classic()+
  theme(plot.title=element_text(size=10,face='bold',margin=margin(10,10,10,0), hjust = 0.5))+
  labs(x='Tree height / m', y='Density')+
  theme(legend.title = element_text(colour="black", size=12, face="bold"))

### TREE CROWN AREA DENSITY PLOT

ggplot()+  
  geom_density(data = par_dfs@data, aes(x=area_sqm, color=Site))+
  geom_density(data = sep_east_dfs@data, aes(x=area_sqm, color=Site))+
  geom_density(data = sep_west_dfs@data, aes(x=area_sqm, color=Site))+
  geom_density(data = dan_dfs@data, aes(x=area_sqm, color=Site))+
  ggtitle('Distribution of crown area per site')+
  theme_classic()+
  theme(plot.title=element_text(size=10,face='bold',margin=margin(10,10,10,0), hjust = 0.5))+
  labs(x='Tree crown area / m2', y='Density')+
  theme(legend.title = element_text(colour="black", size=12, face="bold"))

ggsave("C:/Users/sebhi/ai4er/crown_area_density_small_new_area.png")


summary(sep_west_dfs)


## TREE HEIGHT vs. DIAMETER

### need to clear out NAs for Org_H_mx
par_dfs = par_dfs[!is.na(par_dfs$Org_H_mx), ]
dan_dfs = dan_dfs[!is.na(dan_dfs$Org_H_mx), ]
sep_east_dfs = sep_east_dfs[!is.na(sep_east_dfs$Org_H_mx), ]
sep_west_dfs = sep_west_dfs[!is.na(sep_west_dfs$Org_H_mx), ]


ggplot()+  
  geom_point(data = par_dfs@data, aes(Org_H_md, diameter, color=Site), alpha = 0.2)+
  geom_point(data = sep_east_dfs@data, aes(Org_H_md, diameter, color=Site), alpha = 0.3)+
  geom_point(data = sep_west_dfs@data, aes(Org_H_md, diameter, color=Site), alpha = 0.3)+
  geom_point(data = dan_dfs@data, aes(Org_H_md, diameter, color=Site), alpha = 0.3)+
  ggtitle('Tree height against tree diameter')+
  theme_classic()+
  theme(plot.title=element_text(size=10,face='bold',margin=margin(10,10,10,0), hjust = 0.5))+
  #labs(x='Binned tree heights / m', y='Tree count')+
  theme(legend.title = element_text(colour="black", size=12, face="bold"))

ggsave("C:/Users/sebhi/ai4er/tree_count_small.png")

## CARBON DENSITY PLOT

par_df_carbon_summary= par_dfs@data %>% group_by(Site,Org_H_md_round) %>% 
  summarize(carbon=carbon, num_tot=n(),num_died=sum(Chng_H_md <= (Org_H_md*-0.0279)-0.2, na.rm=TRUE),pct_died_per_year=(100*num_died/num_tot)^(1/3))

summary(par_df_carbon_summary)

ggplot()+  
  geom_density(data = par_dfs@data, aes(x = carbon, color=Site))+
  geom_density(data = sep_east_dfs@data, aes(x = carbon, color=Site))+
  geom_density(data = sep_west_dfs@data, aes(x = carbon, color=Site))+
  geom_density(data = dan_dfs@data, aes(x = carbon, color=Site))+
  ggtitle('Distribution of tree carbon per site')+
  theme_classic()+
  theme(plot.title=element_text(size=10,face='bold',margin=margin(10,10,10,0), hjust = 0.5))+
  labs(x='Carbon / kg', y='Density')+
  theme(legend.title = element_text(colour="black", size=12, face="bold"))

ggsave("C:/Users/sebhi/ai4er/tree_height_density_small_four_sites_correct_area.png")

x_wanker = 5

# joint carbon plot


ggplot()+
  geom_point(data=par_dfs@data, aes(Org_H_md, carbon, color=Site, size=area_sqm), alpha=0.3)+
  geom_point(data=sep_east_dfs@data, aes(Org_H_md, carbon, color=Site, size=area_sqm), alpha=0.3)+
  geom_point(data=sep_west_dfs@data, aes(Org_H_md, carbon, color=Site, size=area_sqm), alpha=0.3)+
  geom_point(data=dan_dfs@data, aes(Org_H_md, carbon, color=Site, size=area_sqm), alpha=0.3)+
  ylim(0, 20000)+ggtitle('Carbon stored in each tree predicted by Mask R-CNN')+
  theme_classic()+
  theme(plot.title=element_text(size=10,face='bold',margin=margin(10,10,10,0), hjust = 0.5))+
  labs(x='Tree height / m', y='Carbon / kg')+
  theme(legend.title = element_text(colour="black", size=8, face="bold"))+
  labs(size='Crown Area')+
  guides(size = guide_legend(reverse=TRUE))


ggsave("C:/Users/sebhi/ai4er/carbon_small_four_sites.png")

# JOINT CARBON PLOT WITH SMOOTHING

ggplot()+
  geom_smooth(data=par_dfs@data, aes(Org_H_mx, carbon, color=Site, size=area), alpha=0.3)+
  geom_smooth(data=sep_east_dfs@data, aes(Org_H_mx, carbon, color=Site, size=area), alpha=0.3)+
  geom_smooth(data=sep_west_dfs@data, aes(Org_H_mx, carbon, color=Site, size=area), alpha=0.3)+
  geom_smooth(data=dan_dfs@data, aes(Org_H_mx, carbon, color=Site, size=area), alpha=0.3)+
  #ylim(0, 20000)+
  geom_vline(xintercept=50)+
  geom_hline(yintercept=2200)+
  ggtitle('Carbon stored in each tree predicted by Mask R-CNN')+
  theme_classic()+
  theme(plot.title=element_text(size=10,face='bold',margin=margin(10,10,10,0), hjust = 0.5))+
  labs(x='Tree height / m', y='Carbon / kg')+
  theme(legend.title = element_text(colour="black", size=8, face="bold"))+
  labs(size='Crown Area')+
  guides(size = guide_legend(reverse=TRUE))




# do plots filtering for trees with confidence > 50%

sep_east_dfs_hi_conf = sep_east_dfs[sep_east_dfs$score > 0.9,]
sep_west_dfs_hi_conf = sep_west_dfs[sep_east_dfs$score > 0.9,]
dan_dfs_hi_conf = dan_dfs[dan_dfs$score > 0.9,]
par_dfs_hi_conf = par_dfs[par_dfs$score > 0.9,]

ggplot()+
  geom_point(data=par_dfs_hi_conf@data, aes(Org_H_md, Chng_H_md, color=Site), alpha=0.4)+
  geom_point(data=sep_east_dfs_hi_conf@data, aes(Org_H_md, Chng_H_md, color=Site), alpha=0.5)+
  geom_point(data=sep_west_dfs_hi_conf@data, aes(Org_H_md, Chng_H_md, color=Site), alpha=0.5)+
  geom_point(data=dan_dfs_hi_conf@data, aes(Org_H_md, Chng_H_md, color=Site), alpha=0.2)+
  geom_smooth(method='rlm', data=par_dfs_hi_conf@data, aes(Org_H_md,Chng_H_md), color='darkgreen', alpha=1)+
  geom_smooth(method='rlm', data=sep_east_dfs_hi_conf@data, aes(Org_H_md,Chng_H_md), color='navy', alpha=0.6)+
  geom_smooth(method='rlm', data=sep_west_dfs_hi_conf@data, aes(Org_H_md,Chng_H_md), color='navy', alpha=0.6)+
  geom_smooth(method='rlm', data=dan_dfs_hi_conf@data, aes(Org_H_md,Chng_H_md), color='darkred', alpha=1)+
  geom_hline(yintercept=0, linetype='dashed', size=0.5)+
  ggtitle('Change in tree height per site predicted by Mask R-CNN score > 0.5')+
  theme_classic()+
  theme(plot.title=element_text(size=10,face='bold',margin=margin(10,10,10,0), hjust = 0.5))+
  labs(x='Original tree height / m', y='Change in tree height / m')+
  theme(legend.title = element_text(colour="black", size=8, face="bold"))+
  labs(size='Crown Area')+
  guides(size = guide_legend(reverse=TRUE))+
  xlim(-10,80)+
  ylim(-60, 20)

ggsave("C:/Users/sebhi/ai4er/tree_growth_80_conf_small.png")


## joint mortality 

par_df_hi_conf_summary= par_dfs_hi_conf@data %>% group_by(Site,Org_H_md_round) %>% 
  summarize(num_tot=n(),num_died=sum(Chng_H_md <= (Org_H_md*-0.0279)-0.2, na.rm=TRUE),pct_died_per_year=(100*num_died/num_tot)^(1/3))
sep_df_hi_conf_summary= sep_dfs_hi_conf@data %>% group_by(Site,Org_H_md_round) %>% 
  summarize(num_tot=n(),num_died=sum(Chng_H_md <= (Org_H_md*-0.0456)-2.2, na.rm=TRUE),pct_died_per_year=(100*num_died/num_tot)^(1/6))
dan_df_hi_conf_summary= dan_dfs_hi_conf@data %>% group_by(Site,Org_H_md_round) %>% 
  summarize(num_tot=n(),num_died=sum(Chng_H_md <= (Org_H_md*-0.1049)-11.1, na.rm=TRUE),pct_died_per_year=(100*num_died/num_tot)^(1/6))

ggplot()+
  geom_point(data = par_df_hi_conf_summary, aes(Org_H_md_round, pct_died_per_year, color=Site))+
  geom_point(data = sep_df_hi_conf_summary, aes(Org_H_md_round, pct_died_per_year, color=Site))+
  geom_point(data = dan_df_hi_conf_summary, aes(Org_H_md_round, pct_died_per_year, color=Site))+
  geom_line(data = par_df_hi_conf_summary, aes(Org_H_md_round, pct_died_per_year, color=Site))+
  geom_line(data = sep_df_hi_conf_summary, aes(Org_H_md_round, pct_died_per_year, color=Site))+
  geom_line(data = dan_df_hi_conf_summary, aes(Org_H_md_round, pct_died_per_year, color=Site))+
  geom_errorbar(data = par_df_hi_conf_summary, aes(x=Org_H_md_round, ymin=pct_died_per_year-0.1, ymax=pct_died_per_year+0.1, color=Site), width=.2, position=position_dodge(0.05))+
  geom_errorbar(data = sep_df_hi_conf_summary, aes(x=Org_H_md_round, ymin=pct_died_per_year-0.2, ymax=pct_died_per_year+0.2, color=Site), width=.2, position=position_dodge(0.05))+
  geom_errorbar(data = dan_df_hi_conf_summary, aes(x=Org_H_md_round, ymin=pct_died_per_year-0.12, ymax=pct_died_per_year+0.12, color=Site), width=.2, position=position_dodge(0.05))+
  
  ggtitle('Annual mortality rate of Mask R-CNN trees')+
  theme_classic()+
  
  theme(plot.title=element_text(size=10,face='bold',margin=margin(10,10,10,0), hjust = 0.5))+
  labs(x='Binned tree heights / m', y='Annual tree mortality / %')+
  theme(legend.title = element_text(colour="black", size=12, face="bold"))+
  xlim(10,60)+
  ylim(0, 4)

ggsave("C:/Users/sebhi/ai4er/mortality_small_90_conf.png")


# tree count

ggplot()+  
  geom_col(data = par_df_hi_conf_summary, aes(Org_H_md_round, num_tot, color=Site))+
  geom_col(data = sep_df_hi_conf_summary, aes(Org_H_md_round, num_tot, color=Site))+
  geom_col(data = dan_df_hi_conf_summary, aes(Org_H_md_round, num_tot, color=Site))+
  ggtitle('Tree count in each height bin')+
  theme_classic()+
  theme(plot.title=element_text(size=10,face='bold',margin=margin(10,10,10,0), hjust = 0.5))+
  labs(x='Binned tree heights / m', y='Tree count')+
  theme(legend.title = element_text(colour="black", size=12, face="bold"))

ggsave("C:/Users/sebhi/ai4er/tree_count_small.png")





# joint carbon plot


ggplot()+
  geom_point(data=par_dfs_hi_conf@data, aes(Org_H_md, carbon, color=Site, size=area), alpha=0.3)+
  geom_point(data=sep_dfs_hi_conf@data, aes(Org_H_md, carbon, color=Site, size=area), alpha=0.1)+
  geom_point(data=dan_dfs_hi_conf@data, aes(Org_H_md, carbon, color=Site, size=area), alpha=0.1)+
  ylim(0, 20000)+ggtitle('Carbon stored in each tree predicted by Mask R-CNN')+
  theme_classic()+
  theme(plot.title=element_text(size=10,face='bold',margin=margin(10,10,10,0), hjust = 0.5))+
  labs(x='Tree height / m', y='Carbon / kg')+
  theme(legend.title = element_text(colour="black", size=8, face="bold"))+
  labs(size='Crown Area')+
  guides(size = guide_legend(reverse=TRUE))


ggsave("C:/Users/sebhi/ai4er/carbon.png")



#### JOINT SCORE PLOT

ggplot()+
  geom_smooth(data=par_dfs@data, aes(Org_H_md, score, color=Site, size=area), alpha=0.3)+
  geom_smooth(data=sep_east_dfs@data, aes(Org_H_md, score, color=Site, size=area), alpha=0.3)+
  geom_smooth(data=sep_west_dfs@data, aes(Org_H_md, score, color=Site, size=area), alpha=0.3)+
  geom_smooth(data=dan_dfs@data, aes(Org_H_md, score, color=Site, size=area), alpha=0.3)+
  #ylim(0, 20000)+
  #geom_vline(xintercept=50)+
  #geom_hline(yintercept=2200)+
  ggtitle('Model confidence with different tree heights')+
  theme_classic()+
  theme(plot.title=element_text(size=10,face='bold',margin=margin(10,10,10,0), hjust = 0.5))+
  labs(x='Tree height / m', y='Confidence')+
  theme(legend.title = element_text(colour="black", size=8, face="bold"))+
  labs(size='Crown Area')+
  guides(size = guide_legend(reverse=TRUE))






#############################
# TOTAL AREA COVERED BY TREES
#############################

summary(sep_east_dfs)

sep_east_sum_area_trees = sum(sep_east_dfs$area, na.rm=TRUE)
sep_west_sum_area_trees = sum(sep_west_dfs$area, na.rm=TRUE)
par_sum_area_trees = sum(par_dfs$area, na.rm=TRUE)
dan_sum_area_trees = sum(dan_dfs$area, na.rm=TRUE)

sep_east_sum_area_trees = sum(sep_east_dfs$area_sqm, na.rm=TRUE)
sep_west_sum_area_trees = sum(sep_west_dfs$area_sqm, na.rm=TRUE)
par_sum_area_trees = sum(par_dfs$area_sqm, na.rm=TRUE)
dan_sum_area_trees = sum(dan_dfs$area_sqm, na.rm=TRUE)

dan_area_site = 234.2890
par_area_site = 1015.8076
sep_east_area_site = 50.5612
sep_west_area_site = 81.1423

dan_tree_coverage = 100 * dan_sum_area_trees/(dan_area_site * 10000)
par_tree_coverage = 100 * par_sum_area_trees/(par_area_site * 10000)
sep_east_tree_coverage = 100 * sep_east_sum_area_trees/(sep_east_area_site * 10000)
sep_west_tree_coverage = 100 * sep_west_sum_area_trees/(sep_west_area_site * 10000)


## Try to calculate areas again

sep_west_dfs$area_sqm <- raster::area(sep_west_dfs) 
sep_east_dfs$area_sqm <- raster::area(sep_east_dfs) 
dan_dfs$area_sqm <- raster::area(dan_dfs) 
par_dfs$area_sqm <- raster::area(par_dfs) 



summary(sep_west_dfs)

crs(sep_west_dfs)
sep_west_dfs$area_sqm <- raster::area(sep_west_dfs) 

sep_west_dfs$area

library(terra)

