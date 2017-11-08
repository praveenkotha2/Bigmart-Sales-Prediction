library(plyr)
library(dplyr)
library(rpart)
library(ggplot2)

traindata <- read.csv("TrainBigmart.csv")
testdata <- read.csv("TestBigmart.csv")

dim(traindata)
dim(testdata)

str(traindata)

summary(traindata)

table(is.na(traindata))
colSums(is.na(traindata))

testdata$Item_Outlet_Sales <- 1
total <- rbind(traindata,testdata)
View(total)

a <- ggplot(total,aes(Item_Type,Item_Weight))+
  geom_boxplot()+
  theme(axis.text.x = element_text(angle = 70, vjust = 0.5, color = "black")) + 
  xlab("Item Type") + 
  ylab("Item Weight") + 
  ggtitle("Item Weight vs Item Type")
a

b <- ggplot(total, aes(Outlet_Identifier, Item_Weight)) +
  geom_boxplot() +
  theme(axis.text.x = element_text(angle = 70, vjust = 0.5, color = "black")) + 
  xlab("Outlet_Identifier") + 
  ylab("Item Weight") + 
  ggtitle("Item Weight vs Outlet identifier")
b

str(total)
weightsByItem <- as.data.frame( ddply(na.omit(total), 
                                      ~Item_Identifier, 
                                      summarise, 
                                      mean=mean(Item_Weight), 
                                      sd=sd(Item_Weight)))


total$Item_Weight <- ifelse(is.na(total$Item_Weight), 
                            weightsByItem$mean[
                              match(total$Item_Identifier, weightsByItem$Item_Identifier)], total$Item_Weight)


c<- ggplot(total, aes(Outlet_Identifier, Item_Weight)) +
  geom_boxplot() +
  theme(axis.text.x = element_text(angle = 70, vjust = 0.5, color = "black")) + 
  xlab("Outlet_Identifier") + 
  ylab("Item Weight") + 
  ggtitle("Item Weight vs Outlet identifier")
c

total$Item_Fat_Content <- revalue(total$Item_Fat_Content,c("LF"="Low Fat","low fat"="Low Fat","reg"="Regular"))

levels(total$Item_Fat_Content) <-c(levels(total$Item_Fat_Content),"None")
total[which(total$Item_Type=="Health and Hygiene"),]$Item_Fat_Content <- "None"
total[which(total$Item_Type=="Household"),]$Item_Fat_Content <- "None"
total[which(total$Item_Type=="Others"),]$Item_Fat_Content <- "None"
table(total$Item_Fat_Content)


VisibilityByItem <- as.data.frame( ddply(na.omit(total), 
                                      ~Item_Identifier, 
                                      summarise, 
                                      mean=mean(Item_Visibility), 
                                      sd=sd(Item_Visibility)))


total$Item_Visibility <- ifelse(total$Item_Visibility==0, 
                            VisibilityByItem$mean[
                              match(total$Item_Identifier, VisibilityByItem$Item_Identifier)], total$Item_Visibility)

summary(total$Item_Visibility)
table(total$Item_Visibility==0)


levels(total$Outlet_Size)[1] <-"Other"
table(total$Outlet_Size)

total$Outlet_Establishment_Year <- 2013 - total$Outlet_Establishment_Year

d <- ggplot(total, aes(x=Item_MRP)) +
  geom_density(color = "blue", adjust=1/5)+
  geom_vline(xintercept = 69, color="red")+
  geom_vline(xintercept = 136, color="red")+
  geom_vline(xintercept = 203, color="red") + 
  ggtitle("Density of Item MRP")
d

total$MRP_Level <- as.factor(
  ifelse(total$Item_MRP < 69, "Low",
         ifelse(total$Item_MRP < 136, "Medium",
                ifelse(total$Item_MRP < 203, "High", "Very_High")))
)

total <- select( total, c(Item_Identifier,
                          Item_Weight,
                          Item_Fat_Content,
                          Item_Visibility,
                          Item_Type,
                          Item_MRP,
                          Outlet_Identifier,
                          Outlet_Size,
                          Outlet_Location_Type,
                          Outlet_Type,
                          Outlet_Establishment_Year,
                          MRP_Level,
   -                       Item_Outlet_Sales
))

e <- ggplot(total[1:nrow(traindata),], aes(Outlet_Identifier, Item_Outlet_Sales)) +
  geom_boxplot() +
  theme(axis.text.x = element_text(angle = 70, vjust = 0.5, color = "black")) + 
  xlab("Outlet identifier") + 
  ylab("Sales") + 
  ggtitle("Sales vs Outlet identifier")
e

total[ which(total$Outlet_Identifier == "OUT010") ,]$Outlet_Size <- "Small"
total[ which(total$Outlet_Identifier == "OUT017") ,]$Outlet_Size <- "Small"
total[ which(total$Outlet_Identifier == "OUT045") ,]$Outlet_Size <- "Small"

total$Item_Identifier <- strtrim(total$Item_Identifier, 3)
total$Item_Identifier <- factor(total$Item_Identifier)
View(total)
str(total)

new_traindata <- total[1:nrow(traindata),]
new_testdata <- total[-(1:nrow(traindata)),]

install.packages(c("e1071","caret","doSNOW","ipred","xgboost"))
library(caret)
library(doSNOW)

train.control1 <- trainControl(method = "repeatedcv",
                              number = 5,
                              repeats = 3,
                              search = "grid")

tune.grid1 <- expand.grid(eta = c(0.05, 0.075, 0.1),
                         nrounds = c(50, 75, 100),
                         max_depth = 6:8,
                         min_child_weight = c(2.0, 2.25, 2.5),
                         colsample_bytree = c(0.3, 0.4, 0.5),
                         gamma = 4,
                         subsample = 1)
View(tune.grid1)

cl1 <- makeCluster(3, type = "SOCK")

registerDoSNOW(cl1)

caret.cv1 <- train(Item_Outlet_Sales ~ ., 
                  data = new_traindata,
                  method = "xgbTree",
                  tuneGrid = tune.grid1,
                  trControl = train.control1)
stopCluster(cl1)

caret.cv1

preds1 <- predict(caret.cv1, new_testdata)
submit <- data.frame(Item_Identifier = testdata$Item_Identifier,Outlet_Identifier=testdata$Outlet_Identifier, Item_Outlet_Sales = preds1)
write.csv(submit, file = "myxgboost3.csv", row.names = FALSE)


library("rpart", lib.loc="C:/Program Files/R/R-3.4.0/library")
treefit <- rpart(Item_Outlet_Sales ~ .,data=new_traindata, 
                 method = "anova",
                 control=rpart.control(minsplit=2, cp=0.01)  )
install.packages("rattle")
install.packages("rpart.plot")
library(rattle)
library(rpart.plot)
library(RColorBrewer)
fancyRpartPlot(treefit)
treeprediction <- predict(treefit,new_testdata)
submit <- data.frame(Item_Identifier = testdata$Item_Identifier,Outlet_Identifier=testdata$Outlet_Identifier, Item_Outlet_Sales = treeprediction)
write.csv(submit, file = "mynew3.csv", row.names = FALSE)
