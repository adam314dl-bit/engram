# Ansible Basics

Автоматизация с Ansible.

## Установка

```bash
pip install ansible
```

## Inventory

```ini
# inventory.ini
[web]
web1.example.com
web2.example.com

[db]
db1.example.com ansible_user=admin

[all:vars]
ansible_python_interpreter=/usr/bin/python3
```

## Ad-hoc команды

```bash
# Ping все хосты
ansible all -i inventory.ini -m ping

# Выполнить команду
ansible web -i inventory.ini -m shell -a "uptime"

# Установить пакет
ansible web -i inventory.ini -m apt -a "name=nginx state=present" --become
```

## Playbook

```yaml
# site.yml
---
- name: Configure web servers
  hosts: web
  become: yes

  vars:
    nginx_port: 80

  tasks:
    - name: Install nginx
      apt:
        name: nginx
        state: present
        update_cache: yes

    - name: Start nginx
      service:
        name: nginx
        state: started
        enabled: yes

    - name: Copy config
      template:
        src: nginx.conf.j2
        dest: /etc/nginx/nginx.conf
      notify: Restart nginx

  handlers:
    - name: Restart nginx
      service:
        name: nginx
        state: restarted
```

```bash
# Запуск
ansible-playbook -i inventory.ini site.yml
```

## Переменные

```yaml
# group_vars/web.yml
nginx_port: 80
app_user: www-data

# host_vars/web1.example.com.yml
nginx_port: 8080
```

## Templates (Jinja2)

```jinja2
# templates/nginx.conf.j2
server {
    listen {{ nginx_port }};
    server_name {{ ansible_hostname }};
}
```
