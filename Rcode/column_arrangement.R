s<-getwd()
substr(s, 1, nchar(s)-5)
datapath<-paste(substr(s, 1, nchar(s)-5),"/data_cleaning/_DeDuplication_On_IDs/deDuplicated_OnID.csv",sep = "")
#If you cannot load the raw dataset, you need to set it by yourself by matching the csv file name.
result <- read.csv(file=datapath,header=T,stringsAsFactors = FALSE)
