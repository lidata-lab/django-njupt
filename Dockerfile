FROM python:3.5
ENV PYTHONUNBUFFERED 1
RUN mkdir /Django
WORKDIR /Django
COPY requirements.txt /Django/
RUN pip install -r requirements.txt \
    && django-admin startproject algorithm
WORKDIR /Django/algorithm
RUN python manage.py startapp urltest \
    && sed -i "s/ALLOWED_HOSTS = \[\]/ALLOWED_HOSTS = \[\'*\'\]/;s/'django.middleware.csrf.CsrfViewMiddleware'/#'django.middleware.csrf.CsrfViewMiddleware'/" /Django/algorithm/algorithm/settings.py \
    && sed -i "s/import path/import path, include/;19 a\    path('urltest/', include('urltest.urls'))," /Django/algorithm/algorithm/urls.py
COPY ./algorithm/ /Django/algorithm/
EXPOSE 8000
ENTRYPOINT ["python", "/Django/algorithm/manage.py", "runserver", "0:8000"]
