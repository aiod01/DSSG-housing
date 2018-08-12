if (!require(dplyr)) install.packages("dplyr")
library("dplyr")


s<-getwd()
datapath1<-paste(s,"/Desktop/DSSG/gitscripper/DSSG-2018_Housing/results/Standardized_Deduped_Datasets/meeting_sample.csv",sep = "")
datapath2<-paste(s,"/Desktop/DSSG/gitscripper/DSSG-2018_Housing/results/Standardized_Deduped_Datasets/1000samples_20180810.csv",sep = "")

dat1 <- read.csv(file=datapath1,header=T,stringsAsFactors = FALSE,na.strings = c("","NA")) %>% select(Category,url)
dat2 <- read.csv(file=datapath2,header=T,stringsAsFactors = FALSE,na.strings = c("","NA"))

total <- left_join(dat2,dat1,by="url")

write.csv(total, file = "/Users/hyeongcheolpark/Desktop/DSSG/gitscripper/DSSG-2018_Housing/results/Standardized_Deduped_Datasets/1000samples_20180810_merged.csv")
