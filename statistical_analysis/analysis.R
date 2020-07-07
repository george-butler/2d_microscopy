library(ggplot2)
library(directlabels)
library(lme4)
library(MuMIn)  
library(lmerTest)
library(grid)
library(gridExtra)
library(lsmeans)
joined<-read.csv("/home/george/Documents/GitHub/final_data.csv")
joined$population_type<-factor(joined$population_type, levels = c("Ancestor","Escape","Invasion","Colonisation"))
######################################################################################################################################
curv_shape_speed_re = lmer(log(curv_moment_speed) ~ population_type +(1|uni_id), data = joined)
anova(curv_shape_speed_re)
mult_comp<-lsmeans(curv_shape_speed_re,list(pairwise~population_type),adjust = "bonferroni")

ordered_level<-levels(joined$population_type)
mean_curv_shape_values = array(data = NA, c(0,2))
for ( i in 1:length(ordered_level)){
  j1<-joined
  j1$population_type<-relevel(j1$population_type, i)
  curve_shape_speed_re = lmer(log(curv_moment_speed) ~ population_type + (1|uni_id), data = j1, REML = TRUE)
  model_sum<-summary(curve_shape_speed_re)
  model_coef<-model_sum[["coefficients"]]
  intercept_val<-model_coef[1,c(1:2)]
  mean_curv_shape_values = rbind(mean_curv_shape_values, intercept_val)
}
mean_curv_shape_values = cbind(mean_curv_shape_values, ordered_level)
mean_curv_shape_values<-as.data.frame(mean_curv_shape_values)
colnames(mean_curv_shape_values)<-c("mean","standard_error","population")
mean_curv_shape_values[] <- lapply(mean_curv_shape_values, as.character)
mean_curv_shape_values$mean<-as.numeric(mean_curv_shape_values$mean)
mean_curv_shape_values$standard_error<-as.numeric(mean_curv_shape_values$standard_error)


joined$mean_curv_moment_speed<-NA
joined$se_curv_moment_speed<-NA

for ( i in 1:length(mean_curv_shape_values$population)){
  joined[joined$population_type == mean_curv_shape_values$population[i],]$mean_curv_moment_speed<-mean_curv_shape_values$mean[i]
  joined[joined$population_type == mean_curv_shape_values$population[i],]$se_curv_moment_speed<-mean_curv_shape_values$standard_error[i]
}

sp<-ggplot(joined, aes(x = population_type, y = log(curv_moment_speed), color=factor(population_type)))+
  geom_errorbar(aes(ymax = mean_curv_moment_speed + 1.96*se_curv_moment_speed, ymin = mean_curv_moment_speed - 1.96*se_curv_moment_speed),width=0.1)+
  geom_point(aes(y=mean_curv_moment_speed), size = 4)+
  scale_x_discrete(labels = c("Ancestor","Escape","Invasion","Colonisation"))+
  scale_color_manual(values = c("#756bb1","#e6550d","darkgreen","#3182bd"),
                     label = c("Ancestor","Escape","Invasion","Colonisation"))+
  ylab("Log(Predictied rate of morphological change)")+xlab("Population")+labs(color = "Population")+
  scale_y_continuous(breaks = seq(4.5,5,0.1),limits = c(4.5,5))+
  theme(axis.text = element_text(size = 18),
        axis.title = element_text(size = 20),
        legend.text = element_text(size = 18),
        legend.title = element_text(size = 20),
        legend.position = "none")

sp
t <- r.squaredGLMM(curv_shape_speed_re)
print(t)
(t[2] - t[1])*100 #as a percentage
####################################################################################################################################
ordered_level<-levels(joined$population_type)
mean_curv_shape_sig = array(data = NA, c(0,(length(ordered_level) + 1)))
for ( i in 1:length(ordered_level)){
  j1<-joined
  j1$population_type<-relevel(j1$population_type, 4)
  curve_shape_speed_re = lmer(log(curv_moment_speed) ~ population_type + (1|uni_id), data = j1, REML = TRUE)
  model_sum<-summary(curve_shape_speed_re)
  model_coef<-model_sum[["coefficients"]]
  intercept_sig<-model_coef[c(1:4),5]
  holder<-c()
  for (j in 1:length(ordered_level)){
    if ((i+1) == j){
      holder<-append(holder,1)
    }
    holder<-append(holder,intercept_sig[j])
  }
  if ((i == length(ordered_level)) & (j == length(ordered_level))){
    holder<-append(holder,1)
  }
  mean_curv_shape_sig = rbind(mean_curv_shape_sig, holder)
}
mean_curv_shape_sig<-as.data.frame(mean_curv_shape_sig)
colnames(mean_curv_shape_sig)<-c("0",ordered_level)
rownames(mean_curv_shape_sig)<-ordered_level
mean_curv_shape_sig<-t(mean_curv_shape_sig)
####################################################################################################################################
null_shape_model = lmer(log(curv_moment_speed) ~ (1|uni_id)*population_type, data= joined, REML = FALSE)
neighbour_shape_model = lmer(log(curv_moment_speed) ~ log(neigh_min_dist) + (1|uni_id) * population_type, data = joined, REML = FALSE)
tracked_shape_model = lmer(log(curv_moment_speed) ~ (log(curv_tracked_speed) + (1|uni_id))*population_type, data = joined, REML = FALSE)
tracked_neighbour_shape_model = lmer(log(curv_moment_speed) ~ (log(curv_tracked_speed) + log(neigh_min_dist) + (1|uni_id))*population_type, data = joined, REML = FALSE)
interaction_shape_model = lmer(log(curv_moment_speed) ~ (log(curv_tracked_speed) + log(neigh_min_dist) + (log(curv_tracked_speed) * log(neigh_min_dist)) + (1|uni_id)) * population_type, data = joined, REML = FALSE)

anova(null_shape_model)
anova(null_shape_model,tracked_shape_model)
anova(null_shape_model, neighbour_shape_model )

anova(tracked_shape_model, tracked_neighbour_shape_model)

anova(tracked_shape_model, interaction_shape_model)
anova(tracked_neighbour_shape_model, interaction_shape_model)
######################################################################################################################################
final_shape_model = lmer(log(curv_moment_speed) ~ (log(curv_tracked_speed) + log(neigh_min_dist) + log(curv_tracked_speed) * log(neigh_min_dist) + (1|uni_id)) * population_type, data = joined, REML = TRUE)
model_sum<-summary(final_shape_model)
anova(final_shape_model)
ordered_level<-levels(joined$population_type)

tapply(joined$population_type,joined$population_type, length)

model_int<-model_sum[["coefficients"]][c(1,4:6),1]
model_int_corrected<-rep(model_int[1], times=length(model_int))
model_int_corrected<-model_int_corrected + model_int
model_int_corrected[1]<-model_int_corrected[1]/2

model_b1<-model_sum[["coefficients"]][c(2,8:10),1]
model_b1_corrected<-rep(model_b1[1], times=length(model_b1))
model_b1_corrected<-model_b1_corrected + model_b1
model_b1_corrected[1]<-model_b1_corrected[1]/2

model_b2<-model_sum[["coefficients"]][c(3,11:13),1]
model_b2_corrected<-rep(model_b2[1], times=length(model_b2))
model_b2_corrected<-model_b2_corrected + model_b2
model_b2_corrected[1]<-model_b2_corrected[1]/2

model_b3<-model_sum[["coefficients"]][c(7,14:16),1]
model_b3_corrected<-rep(model_b3[1], times=length(model_b3))
model_b3_corrected<-model_b3_corrected + model_b3
model_b3_corrected[1]<-model_b3_corrected[1]/2
temp<-cbind(model_int_corrected,model_b1_corrected,model_b2_corrected,model_b3_corrected)
temp<-as.data.frame(temp)
colnames(temp)<-c("intercept","b1","b2","b3")
temp$group_name<-c(ordered_level)
#####################################################################################################################################
X2_20percent_neighbour<-tapply(log(joined$neigh_min_dist), joined$population_type, quantile, probs = 0.2)
X2_40percent_neighbour<-tapply(log(joined$neigh_min_dist), joined$population_type, quantile, probs = 0.4)
X2_60percent_neighbour<-tapply(log(joined$neigh_min_dist), joined$population_type, quantile, probs = 0.6)
X2_80percent_neighbour<-tapply(log(joined$neigh_min_dist), joined$population_type, quantile, probs = 0.8)
X2_100percent_neighbour<-tapply(log(joined$neigh_min_dist), joined$population_type, max)

X1_20percent_speed<-tapply(log(joined$curv_tracked_speed), joined$population_type, quantile, probs = 0.2)
X1_40percent_speed<-tapply(log(joined$curv_tracked_speed), joined$population_type, quantile, probs = 0.4)
X1_60percent_speed<-tapply(log(joined$curv_tracked_speed), joined$population_type, quantile, probs = 0.6)
X1_80percent_speed<-tapply(log(joined$curv_tracked_speed), joined$population_type, quantile, probs = 0.8)
X1_100percent_speed<-tapply(log(joined$curv_tracked_speed), joined$population_type, max)

joined$int<-NA
joined$b1<-NA
joined$b2<-NA
joined$b3<-NA
joined$x2_20percent_neighbour<-NA
joined$x2_40percent_neighbour<-NA
joined$x2_60percent_neighbour<-NA
joined$x2_80percent_neighbour<-NA
joined$x2_100percent_neighbour<-NA
joined$x1_20percent_speed<-NA
joined$x1_40percent_speed<-NA
joined$x1_60percent_speed<-NA
joined$x1_80percent_speed<-NA
joined$x1_100percent_speed<-NA

for ( i in 1:length(temp$group_name)){
  joined[joined$population_type == temp$group_name[i],]$int<-model_int_corrected[i]
  joined[joined$population_type == temp$group_name[i],]$b1<-model_b1_corrected[i]
  joined[joined$population_type == temp$group_name[i],]$b2<-model_b2_corrected[i]
  joined[joined$population_type == temp$group_name[i],]$b3<-model_b3_corrected[i]
  joined[joined$population_type == temp$group_name[i],]$x2_20percent_neighbour<-X2_20percent_neighbour[i]
  joined[joined$population_type == temp$group_name[i],]$x2_40percent_neighbour<-X2_40percent_neighbour[i]
  joined[joined$population_type == temp$group_name[i],]$x2_60percent_neighbour<-X2_60percent_neighbour[i]
  joined[joined$population_type == temp$group_name[i],]$x2_80percent_neighbour<-X2_80percent_neighbour[i]
  joined[joined$population_type == temp$group_name[i],]$x2_100percent_neighbour<-X2_100percent_neighbour[i]
  joined[joined$population_type == temp$group_name[i],]$x1_20percent_speed<-X1_20percent_speed[i]
  joined[joined$population_type == temp$group_name[i],]$x1_40percent_speed<-X1_40percent_speed[i]
  joined[joined$population_type == temp$group_name[i],]$x1_60percent_speed<-X1_60percent_speed[i]
  joined[joined$population_type == temp$group_name[i],]$x1_80percent_speed<-X1_80percent_speed[i]
  joined[joined$population_type == temp$group_name[i],]$x1_100percent_speed<-X1_100percent_speed[i]
}

joined$pred_y_neighbour_20percent<-joined$int + (log(joined$curv_tracked_speed) * joined$b1) + (joined$b2 * joined$x2_20percent_neighbour) + (joined$b3 * log(joined$curv_tracked_speed) * joined$x2_20percent_neighbour)
joined$pred_y_neighbour_40percent<-joined$int + (log(joined$curv_tracked_speed) * joined$b1) + (joined$b2 * joined$x2_40percent_neighbour) + (joined$b3 * log(joined$curv_tracked_speed) * joined$x2_40percent_neighbour)
joined$pred_y_neighbour_60percent<-joined$int + (log(joined$curv_tracked_speed) * joined$b1) + (joined$b2 * joined$x2_60percent_neighbour) + (joined$b3 * log(joined$curv_tracked_speed) * joined$x2_60percent_neighbour)
joined$pred_y_neighbour_80percent<-joined$int + (log(joined$curv_tracked_speed) * joined$b1) + (joined$b2 * joined$x2_80percent_neighbour) + (joined$b3 * log(joined$curv_tracked_speed) * joined$x2_80percent_neighbour)
joined$pred_y_neighbour_100percent<-joined$int + (log(joined$curv_tracked_speed) * joined$b1) + (joined$b2 * joined$x2_100percent_neighbour) + (joined$b3 *log(joined$curv_tracked_speed) * joined$x2_100percent_neighbour)

joined$pred_y_speed_20percent<-joined$int + (joined$x1_20percent_speed * joined$b1) + (joined$b2 * log(joined$neigh_min_dist)) + (joined$b3 * joined$x1_20percent_speed * log(joined$neigh_min_dist))
joined$pred_y_speed_40percent<-joined$int + (joined$x1_40percent_speed * joined$b1) + (joined$b2 * log(joined$neigh_min_dist)) + (joined$b3 * joined$x1_40percent_speed * log(joined$neigh_min_dist))
joined$pred_y_speed_60percent<-joined$int + (joined$x1_60percent_speed * joined$b1) + (joined$b2 * log(joined$neigh_min_dist)) + (joined$b3 * joined$x1_60percent_speed * log(joined$neigh_min_dist))
joined$pred_y_speed_80percent<-joined$int + (joined$x1_80percent_speed * joined$b1) + (joined$b2 * log(joined$neigh_min_dist)) + (joined$b3 * joined$x1_80percent_speed * log(joined$neigh_min_dist))
joined$pred_y_speed_100percent<-joined$int + (joined$x1_100percent_speed * joined$b1) + (joined$b2 * log(joined$neigh_min_dist)) + (joined$b3 * joined$x1_100percent_speed * log(joined$neigh_min_dist))

a<-joined[joined$population_type == "Ancestor",]
a_m = lm(log(curv_moment_speed) ~ 1, data = a)
a_sum<-summary(a_m)

joined[joined$population_type == "Ancestor",]$pred_y_neighbour_20percent<-a_sum[["coefficients"]][1]
joined[joined$population_type == "Ancestor",]$pred_y_neighbour_40percent<-a_sum[["coefficients"]][1]
joined[joined$population_type == "Ancestor",]$pred_y_neighbour_60percent<-a_sum[["coefficients"]][1]
joined[joined$population_type == "Ancestor",]$pred_y_neighbour_80percent<-a_sum[["coefficients"]][1]
joined[joined$population_type == "Ancestor",]$pred_y_neighbour_100percent<-a_sum[["coefficients"]][1]

e<-joined[joined$population_type == "Escape",]
e_m = lmer(log(curv_moment_speed) ~ log(curv_tracked_speed) + (1|uni_id), data = e, REML = TRUE)
e_sum<-summary(e_m)
joined[joined$population_type == "Escape",]$pred_y_neighbour_20percent<-e_sum[["coefficients"]][1]+(log(joined[joined$population_type == "Escape",]$curv_tracked_speed) * e_sum[["coefficients"]][2])
joined[joined$population_type == "Escape",]$pred_y_neighbour_40percent<-e_sum[["coefficients"]][1]+(log(joined[joined$population_type == "Escape",]$curv_tracked_speed) * e_sum[["coefficients"]][2])
joined[joined$population_type == "Escape",]$pred_y_neighbour_60percent<-e_sum[["coefficients"]][1]+(log(joined[joined$population_type == "Escape",]$curv_tracked_speed) * e_sum[["coefficients"]][2])
joined[joined$population_type == "Escape",]$pred_y_neighbour_80percent<-e_sum[["coefficients"]][1]+(log(joined[joined$population_type == "Escape",]$curv_tracked_speed) * e_sum[["coefficients"]][2])
joined[joined$population_type == "Escape",]$pred_y_neighbour_100percent<-e_sum[["coefficients"]][1]+(log(joined[joined$population_type == "Escape",]$curv_tracked_speed) * e_sum[["coefficients"]][2])

i<-joined[joined$population_type == "Invasion",]
i_m = lmer(log(curv_moment_speed) ~ log(curv_tracked_speed) + (1|uni_id), data = i, REML = TRUE)
i_sum<-summary(i_m)
joined[joined$population_type == "Invasion",]$pred_y_neighbour_20percent<-i_sum[["coefficients"]][1]+(log(joined[joined$population_type == "Invasion",]$curv_tracked_speed) * i_sum[["coefficients"]][2])
joined[joined$population_type == "Invasion",]$pred_y_neighbour_40percent<-i_sum[["coefficients"]][1]+(log(joined[joined$population_type == "Invasion",]$curv_tracked_speed) * i_sum[["coefficients"]][2])
joined[joined$population_type == "Invasion",]$pred_y_neighbour_60percent<-i_sum[["coefficients"]][1]+(log(joined[joined$population_type == "Invasion",]$curv_tracked_speed) * i_sum[["coefficients"]][2])
joined[joined$population_type == "Invasion",]$pred_y_neighbour_80percent<-i_sum[["coefficients"]][1]+(log(joined[joined$population_type == "Invasion",]$curv_tracked_speed) * i_sum[["coefficients"]][2])
joined[joined$population_type == "Invasion",]$pred_y_neighbour_100percent<-i_sum[["coefficients"]][1]+(log(joined[joined$population_type == "Invasion",]$curv_tracked_speed) * i_sum[["coefficients"]][2])



#######################################################################################################################################
library(RColorBrewer)
my_blues = rev(brewer.pal(n = 9, "Blues")[5:9])
my_oranges = "#e6550d"
my_oranges = rep(my_oranges, times = length(my_blues))
my_purples = "#756bb1"
my_purples = rep(my_purples, times = length(my_blues))
my_greens = "darkgreen"
my_greens = rep(my_greens, times = length(my_blues))


joined$percent_category_neighbour<-NA
joined$precent_category_y_prediction_neighbour<-NA
for (i in 1:nrow(joined)){
  temp_value<-log(joined$neigh_min_dist[i])
  if (temp_value <= joined$x2_20percent_neighbour[i]){
    joined$percent_category_neighbour[i]<-20
    joined$precent_category_y_prediction_neighbour[i]<-joined$pred_y_neighbour_20percent[i]
  }
  if ((temp_value > joined$x2_20percent_neighbour[i] ) & (temp_value <= joined$x2_40percent_neighbour[i])){
    joined$percent_category_neighbour[i]<-40
    joined$precent_category_y_prediction_neighbour[i]<-joined$pred_y_neighbour_40percent[i]
  }
  if ((temp_value > joined$x2_40percent_neighbour[i] ) & (temp_value <= joined$x2_60percent_neighbour[i])){
    joined$percent_category_neighbour[i]<-60
    joined$precent_category_y_prediction_neighbour[i]<-joined$pred_y_neighbour_60percent[i]
  }
  if ((temp_value > joined$x2_60percent_neighbour[i] ) & (temp_value <= joined$x2_80percent_neighbour[i])){
    joined$percent_category_neighbour[i]<-80
    joined$precent_category_y_prediction_neighbour[i]<-joined$pred_y_neighbour_80percent[i]
  }
  if (temp_value > joined$x2_80percent_neighbour[i]){
    joined$percent_category_neighbour[i]<-100
    joined$precent_category_y_prediction_neighbour[i]<-joined$pred_y_neighbour_100percent[i]
  }
}


joined$percent_category_speed<-NA
joined$precent_category_y_prediction_speed<-NA
for (i in 1:nrow(joined)){
  temp_value<-log(joined$curv_tracked_speed[i])
  if (temp_value <= joined$x1_20percent_speed[i]){
    joined$percent_category_speed[i]<-20
    joined$precent_category_y_prediction_speed[i]<-joined$pred_y_speed_20percent[i]
  }
  if ((temp_value > joined$x1_20percent_speed[i] ) & (temp_value <= joined$x1_40percent_speed[i])){
    joined$percent_category_speed[i]<-40
    joined$precent_category_y_prediction_speed[i]<-joined$pred_y_speed_40percent[i]
  }
  if ((temp_value > joined$x1_40percent_speed[i] ) & (temp_value <= joined$x1_60percent_speed[i])){
    joined$percent_category_speed[i]<-60
    joined$precent_category_y_prediction_speed[i]<-joined$pred_y_speed_60percent[i]
  }
  if ((temp_value > joined$x1_60percent_speed[i] ) & (temp_value <= joined$x1_80percent[i])){
    joined$percent_category_speed[i]<-80
    joined$precent_category_y_prediction_speed[i]<-joined$pred_y_speed_80percent[i]
  }
  if (temp_value > joined$x1_80percent_speed[i]){
    joined$percent_category_speed[i]<-100
    joined$precent_category_y_prediction_speed[i]<-joined$pred_y_speed_100percent[i]
  }
}

joined$percent_category_speed<-as.factor(joined$percent_category_speed)
joined$percent_category_neighbour<-as.factor(joined$percent_category_neighbour)

joined<-within(joined, colour_id_speed<-factor(population_type:percent_category_speed))
joined<-within(joined, colour_id_neighbour<-factor(population_type:percent_category_neighbour))

j1<-joined[joined$population_type == "Colonisation",]
j1$population_type<-factor(j1$population_type)

col_speed<-ggplot(j1, aes(x = log(curv_tracked_speed), y = precent_category_y_prediction_neighbour, color = factor(percent_category_neighbour)))+
  geom_point(data = j1, aes(x = log(curv_tracked_speed),y=log(curv_moment_speed)))+
  geom_line()+scale_colour_manual(values = my_blues)+ylim(3,7)+xlim(3,7)+
  labs(color = "Nearest neighbour \npercentile") + xlab(expression(paste("Log"[e]*" Speed of migration (",mu,"m/h)"))) + ylab(expression("Log"[e]*" Rate of morphological change"))+
  theme(axis.text = element_text(size = 18),
        axis.title = element_text(size = 20),
        legend.text = element_text(size = 14),
        legend.title = element_text(size = 12),
        strip.text = element_text(size= 18),
        legend.position = c(0.86,0.85),
        legend.background = element_rect(fill = "#ffffffaa", colour = NA))


col_neigh<-ggplot(j1, aes(x = log(neigh_min_dist), y = precent_category_y_prediction_speed, color = factor(percent_category_speed)))+
  geom_point(data = j1, aes(x = log(neigh_min_dist),y=log(curv_moment_speed)))+
  geom_line()+scale_colour_manual(values = my_blues)+ylim(3,7)+xlim(0,7)+
  labs(color = "Speed of migration \npercentile") + xlab(expression(paste("Log"[e]*" Nearest neighbour distance (",mu,"m)"))) + #ylab(expression("Log"[e]*" Rate of morphological change"))+
  theme(axis.text = element_text(size = 18),
        axis.text.y = element_blank(),
        axis.title.y = element_blank(),
        axis.ticks.y = element_blank(),
        axis.title = element_text(size = 20),
        legend.text = element_text(size = 14),
        legend.title = element_text(size = 12),
        strip.text = element_text(size= 18),
        legend.position = c(0.86,0.85),
        legend.background = element_rect(fill = "#ffffffaa", colour = NA))+
  annotate("rect", xmin = 3.979, xmax = 4.896, ymin = 3, ymax = 7, 
           alpha = .2)

g1<-ggplotGrob(col_speed)
g2<-ggplotGrob(col_neigh)

newWidth = unit.pmax(g1$widths[2:3],g2$widths[2:3])

g1$widths[2:3] = as.list(newWidth)
g2$widths[2:3] = as.list(newWidth)
grid.arrange(g1,g2,ncol=2)

j2<-joined[joined$population_type != "Colonisation",]  
j2$population_type<-factor(j2$population_type)
levels(j2$population_type)


ggplot(j2, aes(x = log(curv_tracked_speed),y = precent_category_y_prediction_neighbour, color = factor(colour_id_neighbour)))+
  geom_point(data = j2, aes(x=log(curv_tracked_speed), y=log(curv_moment_speed)))+
  geom_line()+scale_colour_manual(values = c(my_purples,my_oranges,my_greens,my_blues))+ylim(3,6.5)+xlim(3,6.5)+
  xlab(expression(paste("Log"[e]*" Speed of migration (",mu,"m/h)"))) + ylab(expression("Log"[e]*" Rate of morphological change"))+
  theme(axis.text = element_text(size = 18),
        axis.title = element_text(size = 20),
        legend.text = element_text(size = 14),
        legend.title = element_text(size = 12),
        strip.text = element_text(size= 18),
        legend.position = "none")+
  #legend.position = c(0.94,0.353),
  #legend.background = element_rect(fill = "#ffffffaa", colour = NA))+
  facet_wrap( ~ population_type)

ggplot(joined, aes(x = log(curv_tracked_speed),y = precent_category_y_prediction_neighbour, color = factor(colour_id_neighbour)))+
  geom_point(data = joined, aes(x=log(curv_tracked_speed), y=log(curv_moment_speed)))+
  geom_line()+scale_colour_manual(values = c(my_purples,my_oranges,my_greens,my_blues))+ylim(3,6.5)+xlim(3,6.5)+
  labs(color = "Nearest neighbour \npercentile") + xlab(expression(paste("Log"[e]*" Speed of migration (",mu,"m/h)"))) + ylab(expression("Log"[e]*" Rate of morphological change"))+
  theme(axis.text = element_text(size = 18),
        axis.title = element_text(size = 20),
        legend.text = element_text(size = 14),
        legend.title = element_text(size = 12),
        strip.text = element_text(size= 18),
        legend.position = "none")+
  #legend.position = c(0.94,0.353),
  #legend.background = element_rect(fill = "#ffffffaa", colour = NA))+
  facet_wrap( ~ population_type)

levels(joined$population_type)
######################################################################################################################################
qt<-quantile(log(j1$neigh_min_dist), probs = seq(0,1,by=0.01))
temp = array(data = NA, c(0,2))
for ( i in 1:101){
  j1$neigh_temp<-log(j1$neigh_min_dist) - qt[i]
  model<-summary(lmer(log(curv_moment_speed) ~ (log(curv_tracked_speed) + neigh_temp + log(curv_tracked_speed) * neigh_temp + (1|uni_id)), data = j1, REML = TRUE))
  neighbour_sig_val<-model[["coefficients"]][2,5]
  log_neighbour_value<-qt[i]
  c<-cbind(log_neighbour_value,neighbour_sig_val)
  temp = rbind(temp,c)
}

