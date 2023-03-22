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
--제약조건이 설정되지 않은 테이블
create table c_emp (
id number,
name varchar2(25),
salary number,
phone varchar2(15),
dept_id number
);

insert into c_emp (id,name) values (1,'김철수');
insert into c_emp (id,name) values (1,'김기철');
delete from c_emp;
select * from c_emp;
--primary key 제약조건 추가
alter table c_emp add primary key(id);
--primary key 제약조건 삭제
alter table c_emp drop primary key;
--제약조건 이름 지정
alter table c_emp add constraint c_emp_id_pk primary key(id);
--사용자가 만든 제약조건 조회
select * from user_constraints where table_name='C_EMP';
insert into c_emp (id,name) values (1,'김철수');
insert into c_emp (id,name) values (1,'김기철');


--테이블 제거
drop table a_emp;
drop table c_emp;
--제약조건 이름 추가
create table c_emp (
id number primary key,
name varchar2(25),
salary number,
phone varchar2(15),
dept_id number
);

select * from user_constraints where table_name='C_EMP';
insert into c_emp (id,name) values (1,'김철수');
insert into c_emp (id,name) values (1,'김기철');

select * from c_emp;

--2. check 제약조건
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


--3. Foreign key ( , PK 제약조건 외래키 다른 테이블의 를 참조)
--테이블 제거
drop table c_emp;
--제약조건 추가
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
--에러 발생
insert into c_emp (id,name,dept_id) values (6,'park',50);
select * from c_emp;

select * from dept;

--4. unique 제약조건
-- primary key : unique( ) + not null( ) 중복안됨 필수입력
-- 테이블 제거
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
--에러 발생
insert into c_emp (id,name) values (2,'kim');
select * from user_constraints where table_name='C_EMP';
insert into c_emp (id) values (3); -- null 입력 가능
insert into c_emp (id) values (4);
select * from c_emp;
--제약조건 삭제
--alter table drop constraint 테이블 제약조건이름
alter table c_emp drop constraint c_emp_name_un;

insert into c_emp (name) values ('kim');
insert into c_emp (name) values ('kim');
insert into c_emp (name) values ('kim');
select * from c_emp;



--create or replace view as select 뷰이름 명령어
create or replace view emp_v as select empno, ename, job, sal, deptno from emp;
select * from emp_v;
drop view emp_v;

--뷰 생성 및 변경
create or replace view test_v
as
 select empno, ename, e.deptno, dname
 from emp e, dept d 
 where e.deptno=d.deptno;
--생성된 뷰는 테이블처럼 사용 가능
select * from test_v; 
-- , 테이블 뷰 목록 확인
select * from tab;
-- ( ) 뷰의 세부 정보 확인 데이터 사전
select * from user_views;

create index c_emp_name_idx on c_emp(name);

drop index c_emp_name_idx;


-- parsing( ) -> ( ) -> 명령어 분석 실행계획 수립 옵티마이저 실행
-- sql developer : F10( ) 실행계획 보기
-- full scan 모든 레코드를 검사
-- (by index rowid) 인덱스를 사용한 검사
-- index unique scan : 유일한 값
-- index range scan : 유일하지 않은 값
select empno,ename from emp where empno=7900;
select empno,ename from emp where ename='박민철';
--인덱스 추가
create index emp_ename_idx on emp(ename);
-- 인덱스를 사용하여 검색
select empno,ename from emp where ename='박민철';
--인덱스 제거
drop index emp_ename_idx;
--인덱스 테스트를 위한 테이블 생성
create table emp3 (
no number,
name varchar2(10),
sal number
);
-- PL/SQL (Procedural Language)
-- 10 테스트용 레코드 만건 입력
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

-- : table access full, cost:104 인덱스를 사용하지 않을 경우 
select * from emp3 where name='shin691' and sal > 200;
--인덱스 추가
create index emp_name_idx on emp3(name,sal);
--index range scan, cost:11
select * from emp3 where name='shin691' and sal > 200;
--인덱스 정보 확인
-- unique index : primary key, unique 제약조건 컬럼에 적용
-- nonunique index 
select * from user_indexes where table_name='EMP3';
-- and , or 복합인덱스는 연산에서는 사용 가능 연산에서는 사용하지 않음
select * from emp3 where name like 'park1111%' and sal> 300;
select * from emp3 where name like 'park1111%' or sal> 300;
--primary key는 인덱스가 자동으로 생성됨
create table emp4 (
no number primary key,
name varchar2(10),
sal number
);
select * from user_indexes where table_name='EMP4';

--no primary key 컬럼에 설정
alter table emp3 add constraint emp3_no_pk primary key(no);
select * from user_indexes where table_name='EMP3';
--인덱스를 사용할 경우 자동 정렬
select * from emp3 where no>900000;
--primary key ( ) 제약조건 제거 인덱스 제거 
alter table emp3 drop constraint emp3_no_pk;
select * from user_indexes where table_name='EMP3';
--인덱스를 사용하지 않을 경우 자동 정렬이 되지 않음
select * from emp3 where no>90000;