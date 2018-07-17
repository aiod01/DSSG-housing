install.packages("stringdist")
install.packages("PASWR")
install.packages("DescTools")
library(DescTools)
library (MASS)
library(dplyr)
library("stringdist")
library("PASWR")

x <- "I have a pen"
y <- "I have an appen"
StrDist(x, y, method = "normlevenshtein", mismatch = 1, gap = 1, ignore.case = FALSE)



#Load data set
s<-getwd()
substr(s, 1, nchar(s)-5)
datapath<-paste(substr(s, 1, nchar(s)-5),"rental_crawlers/raw_listing.csv",sep = "")
#If you cannot load the raw dataset, you need to set it by yourself by matching the csv file name.
result <- read.csv(file=datapath,header=T,stringsAsFactors = FALSE)

#Arrangnig the dataset by title.
result<- result %>% 
  arrange(description)

#Adding index cuz the rowname is not functioning very well.
result$ID <- seq.int(nrow(result))

result <- result %>% 
  mutate(gcs=paste(lat,long))

#There is no missing values for titles, nor description
#But there are some empty title and description, so I am deleting empty titles.
result <- result[!(result$title==""),]

#CSV file of empty title entry for Jocelyn. 
# empty.title <- result[(result$title==""),]
# write.csv(empty.title, file = "emptytitle.csv", append = FALSE, quote = TRUE, sep = " ",
#             eol = "\n", na = "NA", dec = ".", row.names = TRUE,
#             col.names = TRUE, qmethod = c("escape", "double"),
#             fileEncoding = "")

#Let's delete the exact duplicates from the same name "or" the same description
dif.ttl.or.dif.des<- result %>% 
  filter(!duplicated(title)|!duplicated(description)) %>% 
  arrange(lat,long)%>% 
  select(lat, long, title, description,date:ID)

#same title
same.title <- result %>% 
  filter(duplicated(title))

#same desc
same.desc <- result %>% 
  filter(duplicated(description))

#same title and different description
same.ttl.diff.desc <- same.title %>% 
  filter(!(ID%in%same.desc$ID)) %>% 
  select(lat, long, title, description,date:ID)

#Let A subset that has different title,
#Let B subset that has different description,
#Let c subset that has different location.
#Let's make subset B-A-C:same location with same title, different description
#and name it as temp.

temp <- temp[!is.na(temp$lat),]
temp <- temp[!is.na(temp$long),]

#B-(A∪C)
same.ttl.diff.desc.same.gcs <- temp %>% 
  filter(duplicated(gcs))


#Excluding B-(A∪C)
#same location with same title, different description is excluded.
excl.same.ttl.same.loc.dif.des <- dif.ttl.or.dif.des %>% 
  filter(!(ID%in%same.ttl.diff.desc.same.gcs$ID)) %>% 
  arrange(lat,long)%>%
  select(lat, long, title, description,date:ID)


same.desc.diff.ttl <- same.desc %>% 
  filter(!(ID%in%same.title$ID))%>% 
  select(lat, long, description,date:ID) %>% 
  arrange(description) 

#So many ,,,, description makes it confused, so will delete them.
desc.temp <- gsub(","," ",same.desc.diff.ttl$description)
desc.temp <- gsub("\n"," ",same.desc.diff.ttl$description)
same.desc.diff.ttl<- same.desc.diff.ttl[grep("\\b \\b", desc.temp),] %>% 
  arrange(lat)


#So anyway,if [i,j]value exceeds 400, I will remove it. 
#If i>j, [i,j]and [j,i] will have the same value. I will delete the second one, which means i.
dup.candidates <- c()
for (i in 1:(nrow(edit.matrix)-1)) {
  for (j in (i+1):nrow(edit.matrix)) {
    if (edit.matrix[i,j]<200) {
      dup.candidates <- c(dup.candidates,j)
    }
  }
}

dup.candidates <- dup.candidates[!duplicated(dup.candidates)]



#########For duplicates only
set.seed(1)
##let's sample 100 observations from the A union B.
sampled.index <- sample(1:nrow(dif.ttl.or.dif.des), 100, replace = FALSE)
##
sampled.data <- dif.ttl.or.dif.des %>% 
  filter(as.numeric(rownames(dif.ttl.or.dif.des))%in%sampled.index) %>% 
  arrange(title)

#ID first, removing "" next result goes to the dataset, and we got the matrix.
dif.ttl.or.dif.des <- rownames_to_column(dif.ttl.or.dif.des,var="rowname")
sample.edit.matrix <- matrix(data=NA, nrow=100, ncol=100)
rownames(sample.edit.matrix) <- sampled.data$ID
colnames(sample.edit.matrix) <- sampled.data$ID
for (i in 1:(nrow(sampled.data)-1)) {
  for (j in (i+1):nrow(sampled.data)) {
    first.id <- sampled.data$ID[i]
    second.id <- sampled.data$ID[j]
    matrix.index <- dif.ttl.or.dif.des %>% filter(ID%in%c(first.id,second.id)) %>% dplyr::select(rowname) %>% arrange(rowname)
    matrix.index <- as.numeric(matrix.index$rowname)
    edit.value <- edit.matrix[matrix.index[1],matrix.index[2]]
    sample.edit.matrix[i,j] <- edit.value
    }
  }
sample.edit.vector <- as.vector(sample.edit.matrix)
his.nondup <- hist(sample.edit.vector)
###########

#########So I am comparing the difference of the distributions of two subset.
set.seed(1)
sampled.index <- sample(1:nrow(same.ttl.diff.desc), 100, replace = FALSE)

sampled.data <- same.ttl.diff.desc %>% 
  filter(as.numeric(rownames(same.ttl.diff.desc))%in%sampled.index) %>% 
  arrange(title)

#ID first, removing "" next result goes to the dataset, and we got the matrix.
dif.ttl.or.dif.des <- rownames_to_column(dif.ttl.or.dif.des,var="rowname")
sample.edit.matrix <- matrix(data=NA, nrow=100, ncol=100)
rownames(sample.edit.matrix) <- sampled.data$ID
colnames(sample.edit.matrix) <- sampled.data$ID
for (i in 1:(nrow(sampled.data)-1)) {
  for (j in (i+1):nrow(sampled.data)) {
    first.id <- sampled.data$ID[i]
    second.id <- sampled.data$ID[j]
    matrix.index <- dif.ttl.or.dif.des %>% filter(ID%in%c(first.id,second.id)) %>% dplyr::select(rowname) %>% arrange(rowname)
    matrix.index <- as.numeric(matrix.index$rowname)
    edit.value <- edit.matrix[matrix.index[1],matrix.index[2]]
    sample.edit.matrix[i,j] <- edit.value
  }
}
sample.edit.vector.dup <- as.vector(sample.edit.matrix)
his.du <- hist(sample.edit.vector.dup)
hist(sample.edit.vector.dup)
hist(sample.edit.vector)
View(his.nondup)
###########

a <- as.vector(edit.matrix)
hist(a)



combn(letters[1:4], 2)
(m <- combn(10, 5, min))   # minimum value in each combination
mm <- combn(15, 6, function(x) matrix(x, 2, 3))
stopifnot(round(choose(10, 5)) == length(m),
          c(2,3, round(choose(15, 6))) == dim(mm))
a <- combn(1:100,2)
a <- as.matrix(a)
##
#######housing type -> rooms sqft
a <- dif.ttl.or.dif.des
a$housing_type <- gsub("/","",dif.ttl.or.dif.des$housing_type)


private_index <- grep("\\bprivate room\\b", a$housing_type)
a$rooms <- NA
a <- a %>% 
  filter(!row.names(a)%in%private_index) %>% 
  mutate(rooms=substr(a$housing_type,1,4))
substr(a$housing_type,1,4)
#######
##


# excl.same.desc.same.loc.dif.ttl<- excl.same.ttl.same.loc.dif.des%>%
#   filter(!ID%in%temp.2$ID) %>% 
#   arrange(ID)
# 
# a <- excl.same.desc.same.loc.dif.ttl
# 
# a <- a[!is.na(a$lat),] 
# a <- a[!is.na(a$price),]
# a <- a[!is.na(a$location),]
# a <- a %>% 
#   filter(duplicated(lat)&duplicated(long)&duplicated(location)&duplicated(price))
# x <- c(1,2,3,NA,4)
# is.na(x)
# na.omit(x)


# arranged.result <- result %>% 
#   arrange(title)
# 
# arranged.title=data.frame(arranged.result$title)
# names(arranged.title)[names(arranged.title)=="arranged.result.title"]="title1"
# arranged.title$title1=as.character(arranged.title$title1)
# arranged.title$title2 <- ""
# 
# 
# start.time <- Sys.time()
# for (i in (nrow(arranged.title)-1)) {
#     x <- agrep(arranged.title$title1[i],arranged.title$title1[-(1:i)],ignore.case=TRUE, value=TRUE)
#     x <- paste0(x,"")
#     arranged.title$title2[i] <- x
#     }
#How am I gonna get the exact distance ? two output theshold and disatnce. 
#simluation for minimizing 
#prob model CV. prob matching messey data 
# end.time <- Sys.time()
# time.taken <- end.time - start.time
# time.taken
# 
# sum(is.na(arranged.title$title2))


# c <- stringdistmatrix(temp$description,temp$description)
# candidates <- c()
# for (i in 1:(nrow(temp)-1)) {
#   for (j in (i+1):nrow(temp)) {
#     if (c[i,j]<200) {
#       candidates <- c(candidates,j)
#     }
#   }
# }
# candidates <- candidates[!duplicated(candidates)]
# # for (i in (nrow(arranged.title)-1)) {
# #   x <- agrep(arranged.title$title1[i],arranged.title$title1[-(1:i)],ignore.case=TRUE, value=TRUE)
# #   x <- paste0(x,"")
# #   arranged.title$title2[i] <- x
# # }
# 
# 
# 
# start.time <- Sys.time()
# b <- stringdistmatrix(dif.ttl.or.dif.des$description,dif.ttl.or.dif.des$description)
# end.time <- Sys.time()
# time.taken <- end.time - start.time
# time.taken

