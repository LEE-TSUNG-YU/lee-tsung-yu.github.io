library(TSA)
library(tseries)
library(MASS)
library(fGarch)
library(nortest)
trash <- read.csv("D:/Desktop/成大作業/大三上/時間數列分析/第11組 李宗祐、梁鈞翔、林緯鴻/資料/15years_trash.csv",header = T, sep = ",")
trash2 <- as.numeric(trash$trash)
trash_by_month = ts(trash2 , start = c(2001,1) , frequency = 12)

plot(trash_by_month, type = "o" , main = "\"trash\" time series plot" , ylab = "trash")
plot(trash_by_month, type = "l" ,main = "\"trash\" time series plot with monthly symbols" , ylab = "trash")
month. = c("1","2","3","4","5","6","7","8","9","O","N","D")
points(trash_by_month,pch=as.vector(month.))
monthplot(trash_by_month , main = "trash's month plot")

BoxCox.ar(trash_by_month,  lambda=seq(-4, 2, 0.05)) # mle = -1.55
plot(log(trash_by_month), type = "l" , main = "\"log(trash)\" time series plot" , ylab = "log(trash)")
adf.test(log(trash_by_month)) # reject H0
kpss.test(log(trash_by_month)) # reject H0
acf(log(trash_by_month) , main = "ACF of log(trash)" , lag.max = 72)
pacf(log(trash_by_month) , main = "PACF of log(trash)" , lag.max = 60)

trash_diff = diff(log(trash_by_month))
plot(diff(log(trash_by_month)), type = "l" , main = "diff(log(trash))")
acf(diff(log(trash_by_month)), lag.max = 72 , main = "ACF of diff(log(trash))")# seems have seasonal effect and MA(1)
pacf(diff(log(trash_by_month)),lag.max = 72 , main = "PACF of diff(log(trash))")# seems have SAR(2) and MA(1)
eacf(diff(log(trash_by_month)))
# difference once + difference seasonal(12)
trash_diff_diff_seasonal = diff(diff(log(trash_by_month) , lag = 12))
plot(diff(diff(log(trash_by_month) , lag = 12)) , type = "l",main = "diff(diff(log(trash_by_month) , lag = 12))")
month. = c("1","2","3","4","5","6","7","8","9","O","N","D")
points(diff(diff(log(trash_by_month) , lag = 12)),pch=as.vector(month.))
acf(diff(diff(log(trash_by_month) , lag = 12)) , lag = 72 ,main = "acf of diff(diff(log(trash_by_month) , lag = 12)) ")
pacf(diff(diff(log(trash_by_month) , lag = 12)) , lag = 72,main = "pacf of diff(diff(log(trash_by_month) , lag = 12)) ")
eacf(diff(diff(log(trash_by_month) , lag = 12)))
adf.test(diff(diff(trash_by_month , lag = 12))) 
kpss.test(diff(diff(trash_by_month , lag = 12)))

detect_outlier = function(x){
  detectAO(x)
  detectIO(x)
}

m1.1 = arima(log(trash_by_month),order = c(0,1,1),seasonal = list(order=c(2,0,0),period = 12))
m1.1 # AIC = -671.7
# plot(log(trash_by_month) , type = 'o') ; points(fitted(m1.1) , pch = 20 , col = 3)
res1.1 = residuals(m1.1)
plot(res1.1);abline(h=0);t.test(res1.1)
qqnorm(res1.1) ; qqline(res1.1) ; shapiro.test(res1.1) # 0.0001188
acf(res1.1 , lag.max = 60)
Box.test(res1.1 , lag = 12 , type = "Ljung") # 0.3604
Box.test(res1.1 , lag = 16 , type = "Ljung") # 0.08428
Box.test(res1.1 , lag = 48 , type = "Ljung") # 0.02456
Box.test(res1.1 , lag = 60 , type = "Ljung") # 0.01529
detect_outlier(m1.1) # 2,26
# McLeod.Li.test(y = as.numeric(res1.1)) # OK

m1.1.1 = arimax(log(trash_by_month),order = c(0,1,1),seasonal = list(order=c(2,0,0),period = 12)
            ,io = c(2,26)) #,fixed = c(NA,NA,0,0,0,NA,NA,NA),io=c(14,21))
m1.1.1 # AIC = -693.79
res1.1.1 = residuals(m1.1.1)
plot(res1.1.1);abline(h=0)
hist(res1.1.1,main = "");t.test(res1.1.1) # 0.9562
qqnorm(res1.1.1) ; qqline(res1.1.1) ; shapiro.test(res1.1.1) # 0.01079
acf(res1.1.1,lag.max = 60 , main = "")
Box.test(res1.1.1 , lag = 12 , type = "Ljung") # 0.5667
Box.test(res1.1.1 , lag = 16 , type = "Ljung") # 0.1731
Box.test(res1.1.1 , lag = 60 , type = "Ljung") # 0.1527
detect_outlier(m1.1.1) # decide not to add 122 into IO.
# ks.test(res1.1.1 , "pnorm" , mean(res1.1.1),sd(res1.1.1)) # 0.2096
McLeod.Li.test(y = as.numeric(res1.1.1)) # OK

plot(density(scale(res1.1.1)),main = "")
curve(dt(x,df = 179) ,from = -5 , to = 5 , add = T , col = "blue" , lwd = 3)
curve(dcauchy(x , s = 0.625),from = -5 , to = 5 , add = T, col = "red" , lwd = 2)
legend("topright",c("residuals","t(df=179)","cauchy(s=0.625)"),lty=1,lwd = 2,col=c(1,"blue","red"))

plot(density(res1.1.1) , main = "")
curve(dcauchy(x , s = 0.021),from = -0.15 , to = 0.15 , add = T, col = "red" , lwd = 2)
legend("topright",c("residuals","cauchy(s=0.021)"),lty=1,lwd = 2,col=c(1,"red"))
# fitdistr(res1.1.1,"cauchy") 

# m1.1.2 = arimax(log(trash_by_month),order = c(0,1,1),seasonal = list(order=c(2,0,0),period = 12)
               # ,io = c(2,26,122))
# m1.1.2
# res1.1.2 = residuals(m1.1.2)
# shapiro.test(res1.1.2)
# acf(res1.1.2,lag.max = 60)
# Box.test(res1.1.2 , lag = 12 , type = "Ljung") # 0.1713
# Box.test(res1.1.2 , lag = 11 , type = "Ljung") # 0.4876
# Box.test(res1.1.2 , lag = 16 , type = "Ljung") # 0.03909
# Box.test(res1.1.2 , lag = 60 , type = "Ljung") # 0.07914
# Box.test(res1.1.2 , lag = 21 , type = "Ljung") # 0.04388

# m1.2 = arima(log(trash_by_month),order = c(4,1,0),seasonal = list(order=c(2,0,0),period = 12))
# m1.2
# res1.2 = residuals(m1.2)
# plot(res1.2);abline(h=0)
# hist(res1.2,main = "");t.test(res1.2) # 0.7731
# qqnorm(res1.2) ; qqline(res1.2) ; shapiro.test(res1.2) # 0.0002903
# acf(res1.2,lag.max = 60 , main = "")
# Box.test(res1.2 , lag = 12 , type = "Ljung") # 0.3584
# Box.test(res1.2 , lag = 16 , type = "Ljung") # 0.0966
# Box.test(res1.2 , lag = 48 , type = "Ljung") # 0.05837
# Box.test(res1.2 , lag = 60 , type = "Ljung") # 0.04278
# detect_outlier(m1.2) # 2
# McLeod.Li.test(y = as.numeric(res1.2))

# m1.2.1 = arimax(log(trash_by_month),order = c(4,1,0),seasonal = list(order=c(2,0,0),period = 12)
#                ,io = c(2))
# m1.2.1
# res1.2.1 = residuals(m1.2.1)
# plot(res1.2.1);abline(h=0)
# hist(res1.2.1,main = "");t.test(res1.2.1) # 0.9809
# qqnorm(res1.2.1) ; qqline(res1.2.1) ; shapiro.test(res1.2.1) # 0.00418
# acf(res1.2.1,lag.max = 60 , main = "")
# Box.test(res1.2.1 , lag = 10 , type = "Ljung") # 0.5867
# Box.test(res1.2.1 , lag = 16 , type = "Ljung") # 0.116
# Box.test(res1.2.1 , lag = 24 , type = "Ljung") # 0.0794
# Box.test(res1.2.1 , lag = 60 , type = "Ljung") # 0.05381
# detect_outlier(m1.2.1) # 122
# McLeod.Li.test(y = as.numeric(res1.2.1)) # OK

# m1.2.2 = arimax(log(trash_by_month),order = c(4,1,0),seasonal = list(order=c(2,0,0),period = 12)
#                 ,io = c(2,122))
# m1.2.2
# res1.2.2 = residuals(m1.2.2)
# plot(res1.2.2);abline(h=0)
# hist(res1.2.2,main = "");t.test(res1.2.2) # 0.8132
# qqnorm(res1.2.2) ; qqline(res1.2.2) ; shapiro.test(res1.2.2) # 0.0116
# acf(res1.2.2,lag.max = 60 , main = "")
# Box.test(res1.2.2 , lag = 12 , type = "Ljung") # 0.1447
# Box.test(res1.2.2 , lag = 16 , type = "Ljung") # 0.03474
# Box.test(res1.2.2 , lag = 60 , type = "Ljung") # 0.01829
# McLeod.Li.test(y = as.numeric(res1.2.2)) # not OK
# m1.3 = arima(log(trash_by_month),order = c(0,1,1),seasonal = list(order=c(0,1,5),period = 12) , fixed = c(NA,NA,0,0,0,NA))
# m1.3
# res1.3 = residuals(m1.3)
# plot(res1.3);abline(h=0)
# hist(res1.3,main = "");t.test(res1.3) # 0.5386
# qqnorm(res1.3) ; qqline(res1.3) ; shapiro.test(res1.3) # 0.00005111
# acf(res1.3,lag.max = 60 , main = "")
# Box.test(res1.3 , lag = 7 , type = "Ljung") # 0.2754
# Box.test(res1.3 , lag = 8 , type = "Ljung") # 0.1263
# Box.test(res1.3 , lag = 28 , type = "Ljung") # 0.08272
# Box.test(res1.3 , lag = 29 , type = "Ljung") # 0.01867
# Box.test(res1.3 , lag = 36 , type = "Ljung") # 0.00346
# detect_outlier(m1.3) # 14,21 , but in fact 14,21 are not outliers.

# m1.4 = arima(log(trash_by_month),order = c(3,1,0),seasonal = list(order=c(1,1,0),period = 12))
# m1.4 # -612.88
# res1.4 = residuals(m1.4)
# plot(res1.4);abline(h=0)
# hist(res1.4,main = "");t.test(res1.4) # 0.7136
# qqnorm(res1.4) ; qqline(res1.4) ; shapiro.test(res1.4) # 0.0000247
# acf(res1.4,lag.max = 60 , main = "")
# Box.test(res1.4 , lag = 10 , type = "Ljung") # 0.02287
# detect_outlier(m1.4) # 14,21,50 , but in fact 14,21,50 are not outliers.

# m1.5 = arima(log(trash_by_month),order = c(0,1,1),seasonal = list(order=c(1,1,0),period = 12))
# m1.5
# res1.5 = residuals(m1.5)
# plot(res1.5);abline(h=0)
# hist(res1.5,main = "");t.test(res1.5) # 0.6392
# qqnorm(res1.5) ; qqline(res1.5) ; shapiro.test(res1.5) # 0.000004889
# acf(res1.5,lag.max = 60 , main = "")
# Box.test(res1.5 , lag = 10 , type = "Ljung") # 0.06448
# Box.test(res1.5 , lag = 12 , type = "Ljung") # 0.03919
# detect_outlier(m1.5) # 14,21,50 , but in fact 14,21,50 are not outliers.

# ============================================================
people <- read.csv("15years_people.csv", header = T, sep = ",")
people2 <- as.numeric(people$people)
people_by_month = ts(people2 , start = c(2001,1) , frequency = 12)
plot(people_by_month, type = "l" , main = "people time series")
plot(people_by_month, type = "l" , 
     main = "\"people\" time series plot with monthly symbols" , ylab = "trash")
month. = c("1","2","3","4","5","6","7","8","9","O","N","D")
points(people_by_month,pch=as.vector(month.))
monthplot(people_by_month , main = "people's month plot")

people_log = log(people_by_month)
plot(log(people_by_month),type = "o" , main = "log(people)")
adf.test(log(people_by_month)) # reject H0 <0.01
kpss.test(log(people_by_month))# reject H0 <0.01
acf(log(people_by_month),lag.max = 60)

people_diff = diff(log(people_by_month))
plot(diff(log(people_by_month)))
acf(diff(log(people_by_month)),lag.max = 60)
pacf(diff(log(people_by_month)) , lag.max = 60)
eacf(diff(log(people_by_month)))
adf.test(people_diff)
kpss.test(people_diff)

people_diff_diff12 = diff(diff(log(people_by_month)),lag = 12)
plot(diff(diff(log(people_by_month)),lag = 12))
acf(diff(diff(log(people_by_month)),lag = 12),lag.max = 60) #MA(3)+SMA(1)
pacf(diff(diff(log(people_by_month)),lag = 12),lag.max = 60) # AR(2)+SAR(1)
eacf(diff(diff(log(people_by_month)),lag = 12))
adf.test(people_diff_diff12)
kpss.test(people_diff_diff12)

# m2.1=arima(log(people_by_month),order=c(1,1,0),seasonal=list(order=c(1,0,0),period=12))
# m2.1
# res2.1 = residuals(m2.1)
# hist(res2.1);t.test(res2.1)
# qqnorm(res2.1);qqline(res2.1);shapiro.test(res2.1)
# acf(res2.1,lag.max = 60)

# m2.2=arima(log(people_by_month),order=c(1,1,0),seasonal=list(order=c(0,0,1),period=12))
# m2.2
# res2.2 = residuals(m2.2)
# hist(res2.2);t.test(res2.2)
# qqnorm(res2.2);qqline(res2.2);shapiro.test(res2.2)
# acf(res2.2,lag.max = 60)

m2.3=arima(log(people_by_month),order=c(1,1,0),seasonal=list(order=c(2,0,0),period=12))
m2.3
res2.3 = residuals(m2.3)
hist(res2.3);t.test(res2.3)
qqnorm(res2.3);qqline(res2.3);shapiro.test(res2.3)
acf(res2.3,lag.max = 60)

# m2.1.1=arimax(log(people_by_month),order=c(1,1,0),seasonal=list(order=c(1,0,0),period=12)
#           ,xtransf=data.frame(SARS.28=(1*seq(people2)==28),SARS.29=(1*seq(people2)==29)
#                               ,SARS.30=(1*seq(people2)==30),SARS.31=(1*seq(people2)==31),china = (1*seq(people2)==100)),
#           transfer=list(c(0,0),c(0,0),c(0,0),c(0,0),c(0,0)),method='ML')
# m2.1.1
# res2.1.1 = residuals(m2.1.1)
# hist(res2.1.1);t.test(res2.1.1)
# qqnorm(res2.1.1);qqline(res2.1.1);shapiro.test(res2.1.1)
# acf(res2.1.1,lag.max = 60)
# Box.test(res2.1.1,lag = 12 , type ="Ljung") # 0.4783
# Box.test(res2.1.1,lag = 26 , type ="Ljung") # 0.3036
# Box.test(res2.1.1,lag = 28 , type ="Ljung") # 0.1284

# m2.2.1=arimax(log(people_by_month),order=c(1,1,0),seasonal=list(order=c(0,0,1),period=12)
#               ,xtransf=data.frame(SARS.28=(1*seq(people2)==28),SARS.29=(1*seq(people2)==29)
#                                   ,SARS.30=(1*seq(people2)==30),SARS.31=(1*seq(people2)==31),china = (1*seq(people2)==100)),
#               transfer=list(c(0,0),c(0,0),c(0,0),c(0,0),c(0,0)),method='ML')
# m2.2.1
# res2.2.1 = residuals(m2.2.1)
# hist(res2.2.1);t.test(res2.2.1) # 0.2813
# qqnorm(res2.2.1);qqline(res2.2.1);shapiro.test(res2.2.1) # 0.2489
# acf(res2.2.1,lag.max = 60) # not good
# Box.test(res2.2.1,lag = 2 , type ="Ljung") # 0.003658

m2.3.1=arimax(log(people_by_month),order=c(1,1,0),seasonal=list(order=c(2,0,0),period=12)
              ,xtransf=data.frame(SARS.28=(1*seq(people2)==28),SARS.29=(1*seq(people2)==29)
                                  ,SARS.30=(1*seq(people2)==30),SARS.31=(1*seq(people2)==31),china = (1*seq(people2)==100)),
              transfer=list(c(0,0),c(0,0),c(0,0),c(0,0),c(0,0)),method='ML')
m2.3.1
res2.3.1 = residuals(m2.3.1)
hist(res2.3.1);t.test(res2.3.1)
qqnorm(res2.3.1);qqline(res2.3.1);shapiro.test(res2.3.1)
acf(res2.3.1,lag.max = 60)
Box.test(res2.3.1,lag = 24 , type ="Ljung") # 0.7743
Box.test(res2.3.1,lag = 26 , type ="Ljung") # 0.4548
Box.test(res2.3.1,lag = 28 , type ="Ljung") # 0.2042

res1 = residuals(m1.1.1)
res2 = residuals(m2.3.1)
res=ts.intersect(res1, res2)
plot(res,yax.flip = T, main = "Residual", ylab = "Trash")
ccf(res1,res2, main = "res(trash) & res(people)")

res1.=res1[-c(1:7)];length(res1.)
res2_7=res2[-c(174:180)];length(res2_7)

length(res1.);length(res2_7)
plot(x = res1., y = res2_7)

mod = lm(res1.~res2_7)
summary(mod)

mod.1 = lm(res1.~res2_7 -1)
summary(mod.1)
res3 = residuals(mod.1)
plot(res3)
hist(res3);t.test(res3)
qqnorm(res3);qqline(res3);shapiro.test(res3)
acf(res3)
Box.test(res3 , lag = 21 , type = "Ljung-Box") # 0.1174
