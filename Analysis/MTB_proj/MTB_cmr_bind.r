library(tidyverse)
library(rmatio)
library(feather)
library(lme4)
library(lmerTest)
library(car)



data_loc <- '/media/ravinderjit/Data_Drive/Data/MTB_proj/Bind_CMR_mm.mat'

vars <- read.mat(data_loc)
dat <- data.frame(vars)

#Make Factors
dat$sid = as.factor(dat$sid)
dat$TCS = as.factor(dat$TCS)
dat$age = as.factor(dat$age)

str(dat)

p <- ggplot(aes(x=TCS,y=acc),data=dat) + geom_boxplot()
p + xlab('TCS') + ylab('Accuracy')

p <- ggplot(aes(x=age,y=acc,color=TCS),data=dat) + geom_boxplot()
p + xlab('CST') + ylab('Accuracy')

m <- lmer(acc ~ cmr + TCS + (1|sid),data=dat)
summary(m)
Anova(m,test.statistic='F')
anova(m)




