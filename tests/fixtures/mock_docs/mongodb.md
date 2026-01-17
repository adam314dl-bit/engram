# MongoDB

Основы работы с MongoDB.

## Подключение

```bash
mongosh
mongosh "mongodb://localhost:27017/mydb"
```

## Основные команды

```javascript
// Показать базы
show dbs

// Использовать базу
use mydb

// Показать коллекции
show collections
```

## CRUD

### Create

```javascript
// Один документ
db.users.insertOne({
  name: "John",
  email: "john@example.com",
  age: 30
})

// Несколько документов
db.users.insertMany([
  { name: "Alice", age: 25 },
  { name: "Bob", age: 35 }
])
```

### Read

```javascript
// Все документы
db.users.find()

// С условием
db.users.find({ age: { $gt: 25 } })

// Проекция
db.users.find({}, { name: 1, email: 1 })

// Сортировка и лимит
db.users.find().sort({ age: -1 }).limit(10)
```

### Update

```javascript
// Один документ
db.users.updateOne(
  { email: "john@example.com" },
  { $set: { age: 31 } }
)

// Несколько документов
db.users.updateMany(
  { age: { $lt: 30 } },
  { $inc: { age: 1 } }
)
```

### Delete

```javascript
db.users.deleteOne({ email: "john@example.com" })
db.users.deleteMany({ age: { $lt: 18 } })
```

## Индексы

```javascript
// Создание индекса
db.users.createIndex({ email: 1 })

// Составной индекс
db.users.createIndex({ name: 1, age: -1 })

// Текстовый индекс
db.users.createIndex({ bio: "text" })

// Просмотр индексов
db.users.getIndexes()
```

## Агрегация

```javascript
db.orders.aggregate([
  { $match: { status: "completed" } },
  { $group: {
      _id: "$customerId",
      total: { $sum: "$amount" }
  }},
  { $sort: { total: -1 } }
])
```
