# Ansible Roles

Организация кода с ролями.

## Структура роли

```
roles/
└── nginx/
    ├── tasks/
    │   └── main.yml
    ├── handlers/
    │   └── main.yml
    ├── templates/
    │   └── nginx.conf.j2
    ├── files/
    │   └── index.html
    ├── vars/
    │   └── main.yml
    ├── defaults/
    │   └── main.yml
    └── meta/
        └── main.yml
```

## Создание роли

```bash
ansible-galaxy init roles/nginx
```

## Пример роли

```yaml
# roles/nginx/tasks/main.yml
---
- name: Install nginx
  apt:
    name: nginx
    state: present
  tags: nginx

- name: Configure nginx
  template:
    src: nginx.conf.j2
    dest: /etc/nginx/nginx.conf
  notify: Restart nginx

- name: Ensure nginx is running
  service:
    name: nginx
    state: started
    enabled: yes
```

```yaml
# roles/nginx/handlers/main.yml
---
- name: Restart nginx
  service:
    name: nginx
    state: restarted
```

```yaml
# roles/nginx/defaults/main.yml
---
nginx_port: 80
nginx_worker_processes: auto
```

## Использование ролей

```yaml
# site.yml
---
- hosts: web
  become: yes
  roles:
    - nginx
    - { role: app, app_port: 8080 }
    - role: monitoring
      when: enable_monitoring
```

## Ansible Galaxy

```bash
# Установить роль
ansible-galaxy install geerlingguy.docker

# requirements.yml
roles:
  - name: geerlingguy.docker
    version: "6.1.0"

ansible-galaxy install -r requirements.yml
```

## Collections

```bash
# Установить collection
ansible-galaxy collection install community.docker

# Использование
- name: Run container
  community.docker.docker_container:
    name: nginx
    image: nginx:latest
```
