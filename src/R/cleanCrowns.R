library(sf)
library(raster)
library(dplyr)

outputsPath <- "./updated_crowns/paracou_plot_6.shp"

crowns <- st_read(outputsPath)
dim(crowns)
crowns$area <- st_area(crowns)

r <- raster(crs=st_crs(crowns)$proj4string, resolution = c(0.25, 0.25), ext= extent(crowns))

crowns <-
  crowns %>%
    arrange(desc(area))

crowns$ID <- 1:nrow(crowns)
rasCrowns <- rasterize(crowns, r, 'ID', fun='first')
plot(rasCrowns)

getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}


polyid <- extract(rasCrowns, crowns)

crowns$ID2 <- unlist(lapply(polyid, getmode))
crownsFilt <- filter(crowns, ID==ID2 | is.na(ID2))
crownsNeg <- filter(crowns, ID!=ID2)

extent(crowns)[1]

plot(crowns[1], col=rgb(red = 1, green = 0, blue = 0, alpha = 0.5),
     xlim=c(extent(crowns)[1], extent(crowns)[2]),
     ylim=c(extent(crowns)[3],extent(crowns)[4]), axes=TRUE)
plot(crownsFilt[1], col=rgb(red = 1, green = 0, blue = 0, alpha = 0.5),
     xlim=c(extent(crowns)[1], extent(crowns)[2]),
     ylim=c(extent(crowns)[3],extent(crowns)[4]), axes=TRUE)
plot(crownsNeg[1], col=rgb(red = 1, green = 0, blue = 0, alpha = 0.5),
     xlim=c(extent(crowns)[1], extent(crowns)[2]),
     ylim=c(extent(crowns)[3],extent(crowns)[4]), axes=TRUE)

# Reduce number of vertices on output crowns
crownsSimp <- st_simplify(crownsFilt, dTolerance = 0.1)

plot(crownsFilt[1], col=rgb(red = 1, green = 0, blue = 0, alpha = 0.5),
     xlim=c(extent(crowns)[1], extent(crowns)[2]),
     ylim=c(extent(crowns)[3],extent(crowns)[4]), axes=TRUE)
plot(crownsSimp[1], col=rgb(red = 1, green = 0, blue = 0, alpha = 0.5),
     xlim=c(extent(crowns)[1], extent(crowns)[2]),
     ylim=c(extent(crowns)[3],extent(crowns)[4]), axes=TRUE)

st_write(crownsSimp, "./updated_crowns/P6.gpkg", append=FALSE)

