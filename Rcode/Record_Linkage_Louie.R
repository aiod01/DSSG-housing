if (!require(stringdist)) install.packages("stringdist")
if (!require(PASWR)) install.packages("PASWR")
if (!require(DescTools)) install.packages("DescTools")
if (!require(RecordLinkage)) install.packages("RecordLinkage")
library(DescTools)
library (MASS)
library(dplyr)
library(stringdist)
library(PASWR)
library(dplyr)
library(RecordLinkage)


#Load data set
s<-getwd()
substr(s, 1, nchar(s)-5)
datapath<-paste(substr(s, 1, nchar(s)-5),"results/Standardized_Deduped_Datasets/Louie_Clean_20180719.csv",sep = "")
#If you cannot load the raw dataset, you need to set it by yourself by matching the csv file name.
result <- read.csv(file=datapath,header=T,stringsAsFactors = FALSE)

#Arrangnig the dataset by data, delete unnecessary variables.
result<- result %>% 
  arrange(desc(date)) %>% 
  select(-ID,-X)

#Adding index cuz the rowname is not functioning very well.
result$ID <- seq.int(nrow(result))

result <- result %>% 
  mutate(gcs=paste(lat,long))

#There is no missing values for titles, nor gcs
#But there are some empty title and gcs, so I am deleting empty titles.
result <- result[!(result$title==""),]

#Let's delete the exact duplicates from the same name "or" the same gcs
dif.ttl.or.dif.gcs<- result %>% 
  filter(!duplicated(title)|!duplicated(gcs)) 

dif.ttl.and.dif.gcs<- result %>% 
  filter(!duplicated(title)&!duplicated(gcs))


#same title
same.title <- result %>% 
  filter(duplicated(title))

#same desc
same.gcs <- result %>% 
  filter(duplicated(gcs))

#same title and same gcs
same.ttl.same.gcs <- same.title %>% 
  filter((ID%in%same.gcs$ID)) %>% arrange(title)


pairs=compare.dedup(dif.ttl.or.dif.gcs,blockfld=list(c('price','title'),c('sqft','title'),c('gcs','title')))
summary(pairs)



possibles <- getPairs(pairs)
View(possibles)


#Save dif.ttl.or.dif.gcs csv and same.ttl.and.same.csv. 
write.csv(dif.ttl.or.dif.gcs, file = "Louie_Clean_20180726.csv")
write.csv(same.ttl.same.gcs, file = "Known_Duplicated_Louie_20180726.csv")
write.csv(possibles, file = "Candidate_Duplicated_Louie_20180726.csv")
#

                                                                                  