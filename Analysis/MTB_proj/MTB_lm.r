library(tidyverse)
library(rmatio)
library(feather)
library(lme4)
library(lmerTest)
library(car)



dat_loc <- '/media/ravinderjit/Data_Drive/Data/MTB_dataframe'

dat <- read_feather(dat_loc)

#make factors
dat$age = as.factor(dat$age)


m <- lm(CMR ~  memr, data=dat)
summary(m)

ggplot(dat, aes(x = memr, y = CMR)) + 
  geom_point() +
  stat_smooth(method = "lm", col = "red")
