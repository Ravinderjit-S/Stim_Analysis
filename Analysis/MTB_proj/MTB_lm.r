library(tidyverse)
library(rmatio)
library(feather)
library(lme4)
library(car)
library(cocor)



dat_loc <- '/media/ravinderjit/Data_Drive/Data/MTB_dataframe'

dat <- read_feather(dat_loc)

#make factors
dat$age = as.factor(dat$age)



m <- lm(spaced_coh ~ thresh_coh + bind_lapse , data=dat)
summary(m)

svg("Binding_ConsecVsInter.svg")

plot(x=predict(m,dat),dat$spaced_coh)
abline(a=0, b=1)

dev.off()



m2 <- lm(CMR ~ Jane, data=dat)
summary(m2)

plot(x=predict(m2,dat),dat$CMR)
abline(a=0, b=1)


plot(Predict(m),data=dat)



#MTB Phys

m1_phys <- lm(spaced_coh ~  bind_lapse + Ball ,data=dat)
summary(m1_phys)
Anova(m1_phys)


#ggplot(data = dat) + 
#  geom_point(mapping = aes(x = predict(m1_phys,dat), y=spaced_coh)) +
#  geom_abline(intercept=0,slope=1) +
#  labs( x ='Predicted Accuracy', y='True Accuracy', title ='Interrupted Coherence')

svg('Interrupted_vsB.svg')
plot(x=predict(m1_phys,dat),dat$spaced_coh,
     xlab = 'Predicted Interrupted Accuracy', ylab='True Interrupted Accuracy',
     xlim=c(0.3,1),ylim=c(0.3,1), cex.main=1.25, cex.lab=1.25, cex.axis=1.1, lwd=2)
abline(a=0, b=1,lwd=3)
#axis(side=1,lwd=2)
#axis(side=2,lwd=2)
box(lwd=2)
dev.off()


m2_phys <- lm(spaced_coh ~ bind_lapse + Bmn ,data=dat)
summary(m2_phys)

plot(x=predict(m2_phys,dat),dat$spaced_coh)
abline(a=0, b=1)

m2_phys <- lm(spaced_coh ~ bind_lapse + Bmn + Bon ,data=dat)
summary(m2_phys)

plot(x=predict(m2_phys,dat),dat$spaced_coh)
abline(a=0, b=1)

m3_phys <- lm(thresh_coh ~ bind_lapse + Ball + Aall ,data=dat)
summary(m3_phys)

plot(x=predict(m3_phys,dat),dat$thresh_coh)
abline(a=0, b=1)


#MTB beh vs JANE ... not much of relationship

m1_beh <- lm(Jane ~ spaced_coh + thresh_coh + bind_lapse,data=dat)
summary(m1_beh)

#plot(dat$Bon,dat$Jane)

plot(x=predict(m1_beh,dat),dat$Jane)
abline(a=0, b=1)

#MRT vs MTB beh
m2_beh <- lm(MRT ~ spaced_coh:bind_lapse ,dat)
summary(m2_beh)

plot(x=predict(m2_beh,dat),dat$MRT)
abline(a=0, b=1)

#MTB phys vs JANE

m3_phys <- lm(Jane ~ Ball,data=dat)
summary(m3_phys)

svg('MST_Bphys.svg')
plot(x=dat$Ball,dat$Jane,
     xlab = 'Coherent Feature', ylab='MST Threshold',
     cex.main=1.25, cex.lab=1.25, cex.axis=1.1, lwd=2)
abline(m3_phys)
#abline(a=0, b=1,lwd=3)
#axis(side=1,lwd=2)
#axis(side=2,lwd=2)
box(lwd=2)
dev.off()

plot(x=predict(m3_phys,dat),dat$Jane)
abline(a=0, b=1)

#MRT vs MTB
m4_phys <- lm(MRT ~   Ball ,dat)
summary(m4_phys)

svg('MRT_Bphys.svg')

plot(x=dat$Ball,dat$MRT,
     xlab = 'Coherent Feature', ylab='MRT Threshold',
     ylim=c(-6,0), cex.main=1.25, cex.lab=1.25, cex.axis=1.1, lwd=2)
abline(m4_phys)
#abline(a=0, b=1,lwd=3)
#axis(side=1,lwd=2)
#axis(side=2,lwd=2)
box(lwd=2)
dev.off()

#Compare MRT corr with Jane corr

dat2 <-dat %>% filter(MRT!='NA', Jane!='NA')
cocor( ~Ball + MRT | Ball + Jane, dat2)


#MTB phys vs CMR
### add acconting for periphery
m3_beh <- lm(CMR ~ aud_4k + spaced_coh:bind_lapse ,data=dat)
summary(m3_beh)

plot(x=predict(m3_beh,dat),dat$CMR, main='CMR',
     xlab = 'Predicted CMR', ylab='True CMR',
     xlim=c(4,17),ylim=c(4,17), cex.main=1.25, cex.lab=1.25, cex.axis=1.1, lwd=2)
abline(a=0, b=1,lwd=3)
#axis(side=1,lwd=2)
#axis(side=2,lwd=2)
box(lwd=2)

plot(x=predict(m3_beh,dat),dat$CMR)
abline(a=0, b=1)



m5_phys <- lm(CMR ~ aud_4k  + Aall, dat)
summary(m5_phys)

svg('CMR_Aphys.svg')
plot(x=predict(m5_phys,dat),dat$CMR,
     xlab = 'Predicted CMR', ylab='True CMR',
     xlim=c(4,17),ylim=c(4,17), cex.main=1.25, cex.lab=1.25, cex.axis=1.1, lwd=2)
abline(a=0, b=1,lwd=3)
#axis(side=1,lwd=2)
#axis(side=2,lwd=2)
box(lwd=2)

dev.off()

#AQ
m4_beh <- lm(spaced_coh ~ aq + bind_lapse, data=dat)
summary(m4_beh)

plot(x=predict(m4_beh,dat),dat$spaced_coh)
abline(a=0, b=1)

m5_beh <- lm(CMR ~ aq + aud_4k, data=dat)
summary(m4_beh)

plot(x=predict(m5_beh,dat),dat$CMR)
abline(a=0, b=1)

m_aq <- lm(aq ~ Oall*Aall*Ball, data=dat)
summary(m_aq)

plot(x=predict(m_aq,dat),dat$aq)
abline(a=0, b=1)








