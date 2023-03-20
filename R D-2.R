x<-c(0,1,4,9)
y<-c(1,2,3,4)
z<-c(0,5,7,9)
mean(x)
mean(y)
mean(z)

cor(x,y,method="pearson") #기본값
cor(x,y,method="spearman")
cor(y,z,method="pearson")
cor(y,z,method="spearman")
cor(x,z,method="pearson")
cor(x,z,method="spearman")

x<-c(70,72,62,64,71,76,0,65,74,72)
y<-c(70,74,65,68,72,74,61,66,76,75)
cor.test(x,y,method="pearson")
-----------------------------------------------------------------------------
df<-read.csv("c:/data/rides/rides.csv")
head(df)
#산점도
plot(df$overall~df$rides) # y ~ X
# rides overall 와 은 양의 상관관계가 있는 것으로 보임
# (Covariance): , 공분산 두 변수의 상관정도를 나타내는 값 두 변수가 같은 방향으로 움직이는 정도
# x y 의 편차와 의 편차를 곱한 값의 평균값
# X => y => 증가 증가 양수
# X => y => 증가 감소 음수
# 0 공분산이 이면 두 변수는 선형관계가 없음
cov(df$overall, df$rides)
# 양수이므로 양의 상관관계임
# , 공분산은 증가 감소 방향을 이해할 수는 있으나 어느 정도의 상관
관계인지 파악하기는 어려움
cov(1:5, 2:6) # x,y가 같은 방향으로 증가하므로 양수
cov(1:5, rep(3,5)) # x y 0 

cov(1:5, 5:1) # x,y의 증가 방향이 다르므로 음수
cov(c(10,20,30,40,50), 5:1)

#피어슨 상관계수
cor(df$overall, df$rides, method='pearson')
# use='complete.obs' 결측값을 제외하고 계산하는 옵션
cor(df$overall, df$rides, use='complete.obs', method='pearson')

# : 상관계수 검정 상관계수의 통계적 유의성 판단
#통계적으로 유의하다는 것은 관찰된 현상이 전적으로 우연에 의해 
벌어졌을 가능성이 낮다는 의미
# : 0 귀무가설 상관계수가 이다
# : 0 대립가설 상관계수가 이 아니다
cor.test(df$overall, df$rides, method = "pearson", conf.level = 0.95) 
#cor.test(iris$Sepal.Length, iris$Petal.Length, method = 
"pearson", conf.level = 0.95) 
#p-value 0.05 가 이하이므로 귀무가설 기각
# : 결론 두 변수는 선형적으로 상관관계가 있음
# 95% : 0.5252879 0.6407515 신뢰구간
# : 0.5859863 

head(df[,4:8])
#산점도 행렬
plot(df[,4:8])
# ( ) 추세선 회귀선 그리기
pairs(df[,4:8], panel=panel.smooth)
#install.packages("PerformanceAnalytics")
library(PerformanceAnalytics)
chart.Correlation(df[,4:8], histogram=TRUE, pch=19)
# , , 산점도 히스토그램 상관계수가 함께 출력됨
# rides games 0.46 와 
# rides clean 0.79

#결측값이 있는 경우
#df <- na.omit(df)
#상관계수 행렬
cor(df[,4:8])
#상관계수 플롯
#install.packages('corrplot')
library(corrplot)
X<-cor(df[,4:8])
corrplot(X) #원의 크기로 표시됨
#숫자로 출력됨
corrplot(X, method="number")
# method: circle,square,ellipse,number,shade,color,pie
corrplot.mixed(X, lower='ellipse',upper='circle') 
# , addrect 계층적 군집의 결과에 따라 사각형 표시 군집개수
#hclust ; hierarchical clustering order( ) 계층적 군집순서 로 정렬
corrplot(X,order="hclust",addrect=3)

------------------------------------------------------------------------------------------
df <- read.csv("c:/data/ozone/ozone.csv")
#결측값 여부 확인
is.na(df)
#특정 필드의 결측값 확인
is.na(df$Ozone)
#Ozone 필드에 결측값이 있는 행
df[is.na(df$Ozone),]
#결측값의 개수
sum(is.na(df))
#특정 필드의 결측값 개수
sum(is.na(df$Ozone))
#각 샘플의 모든 필드가 NA가 아닐 때 TRUE
#샘플에 결측값이 하나라도 있으면 FALSE
complete.cases(df)
#결측값이 없는 샘플 출력
df[complete.cases(df),]
#결측값이 있는 샘플 출력
df[!complete.cases(df),]
#결측값이 있으므로 계산이 안됨
mean(df$Ozone)
#결측값을 제외하고 계산
mean(df$Ozone, na.rm=T)
#1~2번 필드의 중위수 계산
mapply(median, df[1:2], na.rm=T)
#결측값을 제외
df2<-na.omit(df)
df2
#결측값을 0으로 대체
df3<-df
df3[is.na(df)]<-0
df3
#특정한 필드만 0으로 대체
df4<-df
df4$Ozone[is.na(df4$Ozone)]<-0
df4 
#결측값을 평균값으로 대체
df5<-df
m1<-mean(df[,1], na.rm=T)
m2<-mean(df[,2], na.rm=T)
df5[,1][is.na(df[,1])]<-m1
df5[,2][is.na(df[,2])]<-m2
df5

---------------------------------------------------------------------------------------
# 결측값 시각화 패키지
#install.packages('VIM')
#install.packages('mice')
library(VIM)
library(mice)
win.graph()
md.pattern(df)
#결측값이 없는 샘플 111개
#Ozone 필드에만 결측값이 있는 샘플 35개
#Solar.R 필드에만 결측값이 있는 샘플 5개
#2개 필드에 결측값이 있는 샘플 2개
## 결측값의 개수 표시
win.graph()
#prop=T 백분율로 표시, prop=F 샘플개수로 표시
aggr(df, prop = F, numbers = T)
# 결측값의 위치를 시각적으로 표현(red: 결측값, dark: 빈도수가 높
은 값)
win.graph()
matrixplot(df)


df<-read.csv("c:/data/rides/rides.csv")
head(df)
#범주형 변수는 팩터 자료형으로 변환 후 스케일링 수행
df$weekend <- as.factor(df$weekend)
df$weekend
#install.packages("reshape")
library(reshape)
# melt() 필드 1개를 variable,value 로 여러 행으로 만드는 함수(차원변경)
meltData <- melt(df[2:7])
win.graph()
boxplot(data=meltData, value~variable)
#평균 0, 표준편차 1로 만드는 작업
#스케일링: 표준편차를 1로 만드는 작업
#센터링: 평균을 0으로 만드는 작업
# 정규화된 데이터를 data.frame형태로 변경
df_scaled <- as.data.frame(scale(df[2:7])) #스케일링과 센터링
df_scaled
meltData <- melt(df_scaled)
win.graph()
boxplot(data=meltData, value~variable)
-------------------------------------------------------------------------------------------------------------
#caret 패키지(Classification And Regression Training):분류, 회귀 문제를 풀기 위한 다양한 도구 제공
#install.packages('caret')
library(caret)
df<-read.csv("c:/data/rides/rides.csv")
meltData <- melt(df[2:7])
win.graph()
boxplot(data=meltData, value~variable)
#평균 0, 표준편차 1로 스케일링
prep <- preProcess(df[2:7], c("center", "scale"))
df_scaled2 <- predict(prep, df[2:7])
head(df_scaled2)
meltData <- melt(df_scaled2)
win.graph()
boxplot(data=meltData, value~variable)
#range: 0~1 정규화 
prep <- preProcess(df[2:7], c("range"))
df_scaled3 <- predict(prep, df[2:7])
head(df_scaled3)
meltData <- melt(df_scaled3)
win.graph()
boxplot(data=meltData, value~variable)


========================================================================================
#  이상치 처리
  
df<-read.csv("c:/data/rides/rides.csv")
head(df)
install.packages('car')
library(car)
#회귀분석 모형
model<-lm(overall~num.child + distance + rides + games + 
            wait + clean, data=df)
summary(model)
  #설명력 68.27%
# 1. 아웃라이어 
# 잔차가 2배 이상 크거나 2배 이하로 작은 경우 
outlierTest(model) 
# 이상치 데이터 발견 - 184번 샘플(Bonferonni p value가 0.05보
다 작은 값)
# rstudent - Studentized Residual - 잔차를 잔차의 표준편차로 
나눈 값
# unadjusted p-value : 다중 비교 문제가 있는 p-value
# 본페로니 p - 여러 개의 가설 검정을 수행할 때 다중 비교 문제로 
인해 귀무가설을 기각하게 될 
# 확률이 높아지는 문제를 교정한 p-value
#184번 샘플을 제거한 모형
model2<-lm(overall~num.child + distance + rides + games + 
             wait + clean, data=df[-184,])
model2
summary(model2)
#설명력이 68.27% => 68.76%로 개선됨
#2. 영향 관측치(influential observation) : 모형의 인수들에 불균형한 영향을 미치는 관측치 
# 영향 관측치를 제거하면 더 좋은 모형이 될 수 있음
# Cook's distance(레버리지와 잔차의 크기를 종합하여 영향력을 판단하는 지표)를 이용하여 
# 영향 관측치를 찾을 수 있음
# 레버리지(leverage) : 실제값이 예측값에 미치는 영향을 나타낸 값
# x축: Hat-Values(큰 값은 지렛점)
# y축: Studentized Residuals(표준화 잔차) : 잔차를 표준오차로 나눈 값
win.graph()
influencePlot(model)
#184,103,227,367,373
# 2보다 큰 값, -2보다 작은 값들은 2배 이상 떨어져있는 이상치)
#레버리지와 잔차의 크기가 모두 큰 데이터들은 큰 원으로 표현(영향력이 큰 데이터들)
#184,103,227,367,373
model3=lm(overall~num.child + distance + rides + games + 
            wait + clean, data=df[c(-184,-103,-367,-373),])
model3
summary(model3)
# 설명력 69.12%
===================================상관계수 계산 ========================================
#20명의 신장과 체중 데이터
height <- c(179,166,175,172,173,167,169,172,172,179,161,174,166,176,182,175,177,167,176,177)
weight <- c(113,84,99,103,102,83,85,113,84,99,51,90,77,112,150,128,133,85,112,85)
plot(height,weight)
cor(height,weight)
#기울기와 절편
slope <- cor(height, weight) * (sd(weight) / sd(height))
intercept <- mean(weight) - (slope * mean(height))
slope
intercept
#단순회귀분석 모델 생성
#체중 = 기울기x신장 + 절편
df <- data.frame(height, weight)
df
model <- lm(weight ~ height, data=df) 
#절편(Intercept) -478.816
#기울기 3.347
model
#키가 180인 사람의 체중 예측
model$coefficients[[2]]*180 + model$coefficients[[1]]
summary(model)

plot(height,weight)
abline(model,col='red')
weight
pred<-model$coefficients[[2]]*height + model$coefficients
[[1]]
pred
sum(weight-pred) #오차의 합계는 0
err<-(weight-pred)^2
sum(err) #오차의 제곱합
sum(err/length(weight)) #평균제곱오차(MSE, mean squared error)
#비용함수(cost function) : 평균제곱오차를 구하는 함수
#최적의 가중치(기울기)를 구하기 위한 계산(경사하강법, Gradient 
Descent)
#여기서는 전체의 값이 아닌 1개의 값만 계산
x<-height[1]
y<-weight[1]
w<-seq(-1,2.3,by=0.0001) #가중치, by 간격
#w<-seq(-1,2.3,by=0.1) #가중치, by 간격
pred<-x*w #예측값
err<-(y-pred)^2 #제곱오차
plot(err)
#기울기가 증가하면 오차가 증가하고 기울기가 감소하면 오차가 감소
한다
#기울기가 0에 가까운 값이 최적의 기울기가 된다.
min(err) #최소오차
i<-which.min(err)
paste('최적의 기울기=',w[i])

#최적의 편향(절편)을 구하기 위한 계산
x<-height[1]
y<-weight[1]
w<-0.6313 #가중치
b<-seq(-3.2,3.2,by=0.0001) #편향
#b<-seq(-1,3.2,by=0.1) #편향
pred<-x*w + b #예측값
err<-(y-pred)^2 #제곱오차
plot(err)
#기울기가 증가하면 오차가 증가하고 기울기가 감소하면 오차가 감소한다
#기울기가 0에 가까운 값이 최적의 기울기가 된다.
min(err) #최소오차
i<-which.min(err)
i
paste('최적의 편향=',b[i])

#위의 계산을 통해 얻은 최적의 w,b를 적용한 회귀식
x<-height[1]
y<-weight[1]
w<- 0.6313
b<- -0.00269999999999992
pred<-x*w + b
y
pred
#=========================================단순 회귀 분석 실습2 ============

regression<-read.csv("c:/data/regression/regression.csv",fileEncoding='utf-8')
head(regression)
tail(regression)

summary(regression)
hist(regression$height)
hist(regression$weight)
plot(regression$weight ~ regression$height, main="평균키와 몸무게", xlab="Height", ylab="Weight")
cor(regression$height, regression$weight)
# lm( y ~ x ) x 독립변수, y 종속변수 (x가 한단위 증가할 때 y에게 미치는 영향)
r <- lm(regression$weight ~ regression$height)
plot(regression$weight ~ regression$height, main="평균키와 몸무게", xlab="Height", ylab="Weight")
abline(r,col='red')

#키가 180인 사람의 체중 예측
r$coefficients[[2]]*183 + r$coefficients[[1]]
summary(r)

=======================다중회귀분석 실습=======================
#R에 기본적으로 포함되는 데이터셋 목록
data()
#데이터셋에 대한 도움말
#help(데이터셋이름)
head(attitude)
tail(attitude)
model<-lm(rating ~ . , data=attitude)
model
summary(model)
#complaints, learning이 기여도가 높은 변수
#p-value가 0.05보다 작으므로 통계적으로 유의함
#모델의 설명력(예측의 정확성) 66%
#기여도가 낮은 항목을 제거함으로써 의미있는 회귀식을 구성하는 과정
reduced<-step(model, direction="backward")
#최종적으로 complaints와 learning 2가지 변수 외에는 제거됨
summary(reduced)
#p-value가 0.05보다 작으므로 이 회귀모델은 통계적으로 유의함.
#모델의 설명력(신뢰도,예측정확성) : 68%

=======================다중공선성=============================
#다중공선성(Multicollinearity) : 독립변수끼리 강한 상관관계를 가지는 현상

#다중공선성을 파악하기 위한 수치적 지표
#VIF(Variance Inflation Factor, 분산팽창인자)
# VIFi= 1 / ( 1 - R^2i)
library(car)
#미국 미니애폴리스 지역의 총인구,백인비율,흑인비율,외국태생, 가계소득,
#빈곤,대학졸업비율을 추정한 데이터셋
df<-MplsDemo
head(df)
#독립적인 그래픽창에 그래프 출력
win.graph()
plot(df[,-1])
#독립변수들의 상관계수
cor(df[,2:7])
#install.packages('corrplot')
library(corrplot) 
win.graph()
corrplot(cor(df[,2:7]), method="number")
# white 변수의 경우 다른 변수들과 상관관계가 높음(다중공선성이 의심됨)
model1<-lm(collegeGrad~.-neighborhood,data=df)
summary(model1)
#설명력은 81.86%로 좋은 모형이지만 
#black(흑인비율), foreignBorn(외국태생) 변수의 회귀계수가 
양수로 출력됨
#실제 현상을 잘 설명하지 못하는 모형
#white 변수를 제거한 모형
model2<-lm(collegeGrad~.-neighborhood-white,data=df)
summary(model2)
#설명력은 다소 떨어졌지만 회귀계수가 실제 현상을 잘 설명하는 
것으로 보임
#black(흑인비율)이 음수로 바뀌었음, foreignBorn(외국태생) 
변수는 양수이지만 유의하지 않음
#다중공선성에 대해 확인이 필요한 경우
# p-value가 유의하지 않은 경우
# 회귀계수의 부호가 예상과 다른 경우 
# 데이터를 추가,제거시 회귀계수가 많이 변하는 경우
model<-lm(population~.-collegeGrad-neighborhood,data=df)
# ^{-1} -1승 중괄호를 안써도 됨
print(paste("population의 VIF : ",(1-summary(model)$r.squared)^{-1}))
#다중공선성이 매우 높은 변수
model<-lm(white~.-collegeGrad-neighborhood,data=df)
print(paste("white의 VIF : ",(1-summary(model)$r.squared)^{-1}))
model<-lm(black~.-collegeGrad-neighborhood,data=df)
print(paste("black의 VIF : ",(1-summary(out)$r.squared)^{-1}))
model<-lm(foreignBorn~.-collegeGrad-neighborhood,data=df)
print(paste("foreinBorn의 VIF : ",(1-summary(model)$r.squared)^{-1}))
model<-lm(hhIncome~.-collegeGrad-neighborhood,data=df)
print(paste("hhIncome의 VIF : ",(1-summary(model)$r.squared)^{-1}))
model<-lm(poverty~.-collegeGrad-neighborhood,data=df)
print(paste("poverty의 VIF : ",(1-summary(model)$r.squared)^{-1}))
#다중공선성을 계산해주는 함수
vif(model1)
# 다중공선성이 높은 white 변수 제거
model2<-lm(collegeGrad~.-neighborhood-white,data=df)
summary(model2)
vif(model2)
# vif 수치가 많이 낮아졌고 특히 black의 수치도 많이 낮아졌음

=====================================보스턴 주택 가격 =================================
library(MASS)
head(Boston)
tail(Boston)
dim(Boston)
summary(Boston)
#산점도 행렬
pairs(Boston)

plot(medv~crim, data=Boston, main="범죄율과 주택가격과의 관계", xlab="범죄율", ylab="주택가격")
#범죄율과의 상관계수 행렬 
(corrmatrix <- cor(Boston)[1,]) # 첫번째 변수
#범죄율이 높으면 주택가격이 떨어진다.
#강한 양의 상관관계, 강한 음의 상관관계
corrmatrix[corrmatrix > 0.5 | corrmatrix < -0.5]
#세율과의 상관계수 행렬 
(corrmatrix <- cor(Boston)[10,]) 
#세율이 높으면 주택가격이 떨어진다.
#강한 양의 상관관계, 강한 음의 상관관계
corrmatrix[corrmatrix > 0.5 | corrmatrix < -0.5]
#CHAS: 찰스강의 경계에 위치한 경우는 1, 아니면 0
table(Boston$chas)
#최고가로 팔린 주택들
(seltown <- Boston[Boston$medv == max(Boston$medv),])
#최저가로 팔린 주택들
(seltown <- Boston[Boston$medv == min(Boston$medv),])

#다중회귀분석 모델 생성
(model<-lm(medv ~ . , data=Boston))
#분석결과 요약
summary(model)
#p-value가 0.05보다 작으므로 통계적으로 유의함
#모델의 설명력(예측의 정확성) 73.3%
#전진선택법과 후진제거법
#후진제거법:기여도가 낮은 항목을 제거함으로써 의미있는 회귀식을 
구성하는 과정
reduced<-step(model, direction="backward")
#최종적으로 선택된 변수들 확인
#최종 결과 확인
summary(reduced)
#p-value가 0.05보다 작으므로 이 회귀모델은 통계적으로 유의함.
#모델의 설명력(신뢰도,예측정확성) : 73.4%

=====================주택 가격 예측 ==============================
df<-read.csv("c:/data/house_regress/data.csv")
head(df)
tail(df)
library(dplyr)
# Suburb, Address, Type, Method, SellerG, Date, 
CouncilArea, Regionname필드 제거
df<-df %>% select(-Suburb, -Address, -Type, -Method, 
                  -SellerG, -Date, -CouncilArea, -Regionname)
dim(df)
# 결측값이 있는 행을 제거
df<-na.omit(df)
tail(df)
dim(df)
summary(df)
#상관계수 행렬 
(corrmatrix <- cor(df))
#강한 양의 상관관계, 강한 음의 상관관계
corrmatrix[corrmatrix > 0.5 | corrmatrix < -0.5]
#install.packages("corrplot")
library(corrplot) 
corrplot(cor(df), method="circle")
#다중회귀분석 모델 생성
model<-lm(Price ~ ., data = df )
model
#분석결과 요약
summary(model)
#p-value가 0.05보다 작으므로 통계적으로 유의함
#모델의 설명력(예측의 정확성) 0.4965
#전진선택법과 후진제거법
#후진제거법:기여도가 낮은 항목을 제거함으로써 의미있는 회귀식을 
구성하는 과정
reduced<-step(model, direction="backward")
#최종적으로 선택된 변수들 확인
#최종 결과 확인
summary(reduced)
#p-value가 0.05보다 작으므로 이 회귀모델은 통계적으로 유의함.
#모델의 설명력(신뢰도,예측정확성) : 73.4%
===================회긔분석 모형저장 불러오기=========== 
df<-read.csv("c:/data/rides/rides.csv")
head(df)
model<-lm(overall~num.child + distance + rides + games + 
            wait + clean, data=df)
summary(model)
save(model, file="c:/data/R/rides_regress.model")
rm(list=ls()) #현재 작업중인 모든 변수들을 제거 
load("c:/data/R/rides_regress.model")
ls()
summary(model)  
"-"(2,3)
y=c(1,2,3,NA)
3*y
3^2
c(1,3,5,7)+c(1,2,4,6,8)
s = c("Monday","Tuesday","Wednesday")
substr(s,1,3)
