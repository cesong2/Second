create table c_emp(
id number(5) constraint c_emp_id_pk primary key,
name varchar2(25) ,
salary number(7,2) constraint c_emp_salary_ck 
 check(salary between 100 and 1000),
phone varchar2(15) ,
dept_id number(7) constraint c_emp_dept_id_fk
 references dept(deptno)
);

select constraint_name from user_constraints;
select * from user_constraints where table_name='C_EMP';

alter table c_emp add constraint c_emp_name_un unique(name);

alter table c_emp modify name varchar2(25) not null;

alter table c_emp drop constraint c_emp_name_un;

--primary key
--���������� �������� ���� ���̺�
create table c_emp (
id number,
name varchar2(25),
salary number,
phone varchar2(15),
dept_id number
);

insert into c_emp (id,name) values (1,'��ö��');
insert into c_emp (id,name) values (1,'���ö');
delete from c_emp;
select * from c_emp;
--primary key �������� �߰�
alter table c_emp add primary key(id);
--primary key �������� ����
alter table c_emp drop primary key;
--�������� �̸� ����
alter table c_emp add constraint c_emp_id_pk primary key(id);
--����ڰ� ���� �������� ��ȸ
select * from user_constraints where table_name='C_EMP';
insert into c_emp (id,name) values (1,'��ö��');
insert into c_emp (id,name) values (1,'���ö');


--���̺� ����
drop table a_emp;
drop table c_emp;
--�������� �̸� �߰�
create table c_emp (
id number primary key,
name varchar2(25),
salary number,
phone varchar2(15),
dept_id number
);

select * from user_constraints where table_name='C_EMP';
insert into c_emp (id,name) values (1,'��ö��');
insert into c_emp (id,name) values (1,'���ö');

select * from c_emp;

--2. check ��������
drop table c_emp; 
create table c_emp (
id number(5) ,
name varchar2(25),
salary number(7,2) constraint c_emp_salary_ck
 check(salary between 100 and 1000),
phone varchar2(15),
dept_id number(7)
);
insert into c_emp (id,name,salary) values (1,'kim',500);
insert into c_emp (id,name,salary) values (2,'park',1500);


--3. Foreign key ( , PK �������� �ܷ�Ű �ٸ� ���̺��� �� ����)
--���̺� ����
drop table c_emp;
--�������� �߰�
create table c_emp (
id number primary key,
name varchar2(25),
salary number,
phone varchar2(15),
dept_id number,
foreign key(dept_id) references dept(deptno)
);
insert into c_emp (id,name,dept_id) values (1,'kim',10);
insert into c_emp (id,name,dept_id) values (2,'park',20);
--���� �߻�
insert into c_emp (id,name,dept_id) values (6,'park',50);
select * from c_emp;

select * from dept;

--4. unique ��������
-- primary key : unique( ) + not null( ) �ߺ��ȵ� �ʼ��Է�
-- ���̺� ����
drop table c_emp;
create table c_emp (
id number,
name varchar2(25),
salary number,
phone varchar2(15),
dept_id number,
constraint c_emp_name_un unique(name)
);
insert into c_emp (id,name) values (1,'kim');
--���� �߻�
insert into c_emp (id,name) values (2,'kim');
select * from user_constraints where table_name='C_EMP';
insert into c_emp (id) values (3); -- null �Է� ����
insert into c_emp (id) values (4);
select * from c_emp;
--�������� ����
--alter table drop constraint ���̺� ���������̸�
alter table c_emp drop constraint c_emp_name_un;

insert into c_emp (name) values ('kim');
insert into c_emp (name) values ('kim');
insert into c_emp (name) values ('kim');
select * from c_emp;



--create or replace view as select ���̸� ��ɾ�
create or replace view emp_v as select empno, ename, job, sal, deptno from emp;
select * from emp_v;
drop view emp_v;

--�� ���� �� ����
create or replace view test_v
as
 select empno, ename, e.deptno, dname
 from emp e, dept d 
 where e.deptno=d.deptno;
--������ ��� ���̺�ó�� ��� ����
select * from test_v; 
-- , ���̺� �� ��� Ȯ��
select * from tab;
-- ( ) ���� ���� ���� Ȯ�� ������ ����
select * from user_views;

create index c_emp_name_idx on c_emp(name);

drop index c_emp_name_idx;


-- parsing( ) -> ( ) -> ��ɾ� �м� �����ȹ ���� ��Ƽ������ ����
-- sql developer : F10( ) �����ȹ ����
-- full scan ��� ���ڵ带 �˻�
-- (by index rowid) �ε����� ����� �˻�
-- index unique scan : ������ ��
-- index range scan : �������� ���� ��
select empno,ename from emp where empno=7900;
select empno,ename from emp where ename='�ڹ�ö';
--�ε��� �߰�
create index emp_ename_idx on emp(ename);
-- �ε����� ����Ͽ� �˻�
select empno,ename from emp where ename='�ڹ�ö';
--�ε��� ����
drop index emp_ename_idx;
--�ε��� �׽�Ʈ�� ���� ���̺� ����
create table emp3 (
no number,
name varchar2(10),
sal number
);
-- PL/SQL (Procedural Language)
-- 10 �׽�Ʈ�� ���ڵ� ���� �Է�
declare 
 i number := 1; 
 name varchar(20) := 'kim';
 sal number := 0;
begin
 while i <= 100000 loop
 if i mod 2 = 0 then 
 name := 'kim' || to_char(i);
 sal := 300;
 elsif i mod 3 = 0 then
 name := 'park' || to_char(i);
 sal := 400;
 elsif i mod 5 = 0 then
 name := 'hong' || to_char(i);
 sal := 500;
 else
 name := 'shin' || to_char(i);
 sal := 250;
 end if; 
 insert into emp3 values (i,name,sal); 
 i := i + 1; 
 end loop; 
end;
/ 

-- : table access full, cost:104 �ε����� ������� ���� ��� 
select * from emp3 where name='shin691' and sal > 200;
--�ε��� �߰�
create index emp_name_idx on emp3(name,sal);
--index range scan, cost:11
select * from emp3 where name='shin691' and sal > 200;
--�ε��� ���� Ȯ��
-- unique index : primary key, unique �������� �÷��� ����
-- nonunique index 
select * from user_indexes where table_name='EMP3';
-- and , or �����ε����� ���꿡���� ��� ���� ���꿡���� ������� ����
select * from emp3 where name like 'park1111%' and sal> 300;
select * from emp3 where name like 'park1111%' or sal> 300;
--primary key�� �ε����� �ڵ����� ������
create table emp4 (
no number primary key,
name varchar2(10),
sal number
);
select * from user_indexes where table_name='EMP4';

--no primary key �÷��� ����
alter table emp3 add constraint emp3_no_pk primary key(no);
select * from user_indexes where table_name='EMP3';
--�ε����� ����� ��� �ڵ� ����
select * from emp3 where no>900000;
--primary key ( ) �������� ���� �ε��� ���� 
alter table emp3 drop constraint emp3_no_pk;
select * from user_indexes where table_name='EMP3';
--�ε����� ������� ���� ��� �ڵ� ������ ���� ����
select * from emp3 where no>90000;