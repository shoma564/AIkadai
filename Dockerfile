FROM shomaigu/flask-base:latest

#RUN apt -y install python3 python3-pip curl wget 
#RUN apt autoremove 
#RUN pip3 install --upgrade pip setuptools
#RUN mkdir -p /usr/src/app/templates
RUN mkdir -p /var/www/

ADD ./requirements.txt /usr/src/app/
ADD ./app.py /usr/src/app/
ADD ./wsgi.py /usr/src/app/
ADD ./templates /usr/src/app/templates
ADD ./static /usr/src/app/static
ADD ./uwsgi.ini /usr/src/app/
RUN pip3 install --no-cache-dir -r /usr/src/app/requirements.txt


WORKDIR /usr/src/app/
#CMD ["python3", "/usr/src/app/app.py"]

CMD ["uwsgi","--ini","/usr/src/app/uwsgi.ini"]
