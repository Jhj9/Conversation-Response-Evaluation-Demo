# 단어 레벨 및 문맥 레벨에서의 대화 응답 적절성 자동 평가 프로그램
단어 레벨 및 문맥 레벨에서의 대화 응답 적절성 자동 평가 프로그램

Pycharm

![화면 캡처 2022-04-13 141152](https://user-images.githubusercontent.com/50137851/163105666-d975d5a9-6f46-4015-bd12-92cf726cc163.png)

1) Get from VCS로 시작하기

2) https://github.com/Jhj9/HLILab_Evaluation_Demo_v1.git <br>
입력해서 clone<br>

3) Terminal에<br> 
  pip install django <br>
  pip install rouge <br>
  pip install datasets <br>
  pip install bert_score <br>
입력

4) mysite/secrets.json 생성 후 SECRET_KEY 입력 <br>
  Ex) {"SECRET_KEY": "django-insecure-dml=장고시크릿키50자"} <br>
  https://djecrety.ir/

5) Terminal에  
  python manage.py runserver <br>
입력하면 실행됨

6) Terminal에 뜨는 <br>
  http://127.0.0.1:8000/ <br>
클릭해서 웹페이지 실행

+)
  - 페이지 확대해서 글 중간에 맞추기 <br>
  - 직접입력 처음 실행하면 오래 걸리므로 미리 해 놓기
  - 한글 입력할 시 점수 0점
