-- drop table tr_data_org;
-- drop table tt_data_org;
create table tr_data_org(
PassengerId	bigint comment '乘客编号',
Survived	bigint  comment '存活状况',
Pclass	bigint  comment '客舱等级',
Name	string  comment '名名',
Sex	string  comment '性别',
Age	double  comment '年龄',
SibSp	bigint  comment '同乘兄弟姐妹/配偶数',
Parch	bigint  comment '同乘父母/子女数',
Ticket	string  comment '船票编号',
Fare	double  comment '票价',
Cabin	string  comment '客舱号',
Embarked	string  comment '上船港口')
comment '原始训练数据集';

create table tt_data_org(
PassengerId	bigint comment '乘客编号',
Pclass	bigint  comment '客舱等级',
Name	string  comment '姓名',
Sex	string  comment '性别',
Age	double  comment '年龄',
SibSp	bigint  comment '同乘兄弟姐妹/配偶数',
Parch	bigint  comment '同乘父母/子女数',
Ticket	string  comment '船票编号',
Fare	double  comment '票价',
Cabin	string  comment '客舱号',
Embarked	string  comment '上船港口')
comment '原始测试数据集';

