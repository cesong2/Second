df<-read.csv("c:/data/ozone/ozone2.csv")
head(df)
#############################################################
library(dplyr)
# 필드 제거
df<-df %>% select(-Ozone)
#############################################################
#불균형 데이터셋
tbl<-table(df$Result)
tbl
win.graph(); barplot(tbl, beside = T, legend = T, col =
                       rainbow(2))
#############################################################
# under sampling
#install.packages("ROSE")
library(ROSE)
# method: under,over,both N: 샘플링 후의 샘플 개수(적은 쪽 x2) 또는 p=0.5 50:50으로 선택
df_samp <- ovun.sample(Result ~ . ,data = df, seed=1, method =
                         "under", N=72*2)$data
tbl<-table(df_samp$Result)
tbl
library(caret)
#랜덤 시드 고정
set.seed(123)
#학습용:검증용 8:2로 구분
#list=FALSE, 인덱스값들의 리스트를 반환하지 않음
idx_train <- createDataPartition(y=df_samp$Result, p=0.8,
                                 list=FALSE)
#학습용
train <- df_samp[idx_train, ]
X_train <- train[, -1]
y_train <- train[, 1]
#검증용
test <- df_samp[-idx_train, ]
X_test <- test[, -1]
y_test <- test[, 1]
head(X_train)
head(y_train)
#############################################################
library(rpart)
#최대 깊이를 제한
a<-rpart.control(maxdepth = 20)
#method='class' 범주형(분류모형), method='anova' 연속형(회귀모형)
model <- rpart(Result ~ ., method="class", data=train,
               control=a)
model #트리 모형이 만든 규칙들
# 트리 그래프
win.graph()
plot(model, uniform=T, main="Tree")
text(model, use.n=T, all=T, cex=.8)
#가지치기를 위한 최적의 노드 개수 확인
#CP(Complexity Parameter, 복잡성 매개변수) : 에러율이 가장 낮을 때의 노드 개수
printcp(model)
win.graph(); plotcp(model) # 교차검증 결과
#2개일 때가 최적임
summary(model) # 세부적인 노드 정보
#############################################################
# 트리를 pdf로 저장하기 위한 postscript 생성
# 생성된 ps 파일을 더블클릭한 후 Acrobat Distiller 프로그램을이용하여 pdf 파일을 생성함
post(model, file = "c:/data/ozone/ozone_tree.ps",
     title = "Tree Model")
#############################################################
#가지치기
#에러율이 가장 낮을 때의 CP값으로 가치치기 진행
pfit<- prune(model, cp=model$cptable[which.min(model
                                               $cptable[,"xerror"]),"CP"])
#pfit<-prune(model, cp=model$cptable[3])
win.graph()
plot(pfit, uniform=T, main="Pruned Tree")
text(pfit, use.n=T, all=T, cex=.8)
post(pfit, file = "c:/data/ozone/ozone_ptree.ps", title =
       "Pruned Tree")
#############################################################
#install.packages('rattle')
#트리를 시각적으로 꾸며주는 패키지
library(rattle)
library(rpart.plot)
win.graph(); rpart.plot(pfit) #기본형 트리
win.graph(); fancyRpartPlot(pfit) #파스텔톤의 트리
#############################################################
# predict(트리모델, 검증용데이터셋)
pred <- predict(pfit,newdata=X_test,type='class')
pred
#############################################################
#분류 모델 평가를 위해 오분류표(confusion matrix) 출력
table(y_test, pred)
mean(y_test == pred)
#############################################################
y_test_f<-as.factor(y_test)
confusionMatrix(y_test_f, pred)
