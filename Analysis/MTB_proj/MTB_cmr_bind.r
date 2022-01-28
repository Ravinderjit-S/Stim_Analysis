library(tidyverse)
library(rmatio)
library(feather)
library(lme4)
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


data_loc_bindPhys <- '/media/ravinderjit/Data_Drive/Data/MTB_proj/Bind_phys_mm.mat'
vars_phys <- read.mat(data_loc_bindPhys)
dat_phys <- data.frame(vars_phys)

#Make Factors
dat_phys$sid = as.factor(dat_phys$sid)
dat_phys$type = as.factor(dat_phys$type)
dat_phys$age = as.factor(dat_phys$age)

p <- ggplot(aes(x=type,y=acc),data=dat_phys) + geom_boxplot()
p + xlab('Thresh, Spaced') + ylab('Accuracy')

m_phys <- lmer(acc ~ type + lapse + pca1+ pca2 (1|sid),data=dat_phys)
summary(m_phys)
Anova(m_phys,test.statistic = 'F')
anova(m_phys)

plot(dat_phys$pca1,dat_phys$acc)
abline(m_phys)

#make a lm?
data_loc_lm <- '/media/ravinderjit/Data_Drive/Data/EEGdata/MTB/Model/Binding20TonesModelVars.mat' 
vars_lm <- read.mat(data_loc_lm)

dat_phys_lm <- data.frame(vars_lm$threshCoh,vars_lm$spacedCoh,vars_lm$consecCoh,vars_lm$features[,1],vars_lm$features[,2],vars_lm$pca_feats[,1],vars_lm$pca_feats[,2])
str(dat_phys_lm)

#Make Factors

m_phys_lm <- lm(vars_lm.spacedCoh ~ vars_lm.features...1.,dat=dat_phys_lm)
summary(m_phys_lm)

plot(dat_phys_lm$vars_lm.features...1., dat_phys_lm$vars_lm.spacedCoh)
abline(m_phys_lm)





