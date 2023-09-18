library(tidyverse)
library(rmatio)
library(lme4)
library(car)

data_loc <- '/media/ravinderjit/Data_Drive/Data/MTB_dataframe_2.mat'



dat <- read.mat(data_loc)
dat <- data.frame(dat)

#make Factors
dat$sid = as.factor(dat$sid)
dat$age = as.factor(dat$age)

str(dat)



m <- lm(CMR~ PCA_Ball + aud_4k,data=dat)
summary(m)
Anova(m, test.statistic = 'F')

plot(dat$PCA_Ball, dat$CMR)
abline(m)


m2 <- lmer(acc_inter ~ cond + B_mn1 +B_on1 + (1|sid),data=Inter_data)
summary(m2)
Anova(m2, test.statistic = 'F')

I


