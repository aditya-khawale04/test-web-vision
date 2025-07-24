// import express from 'express'
// const app = express()
// const port = 3000
// // Set EJS View Engine
// app.set('view engine', 'ejs');

// app.get('/', (req, res) => {
//     res.render('index')
// })

// app.listen(port, () => {
//     console.log(`Example app listening on port ${port}`)
// })

const express = require('express');
const app = express();
const port = 3000;

app.set('view engine', 'ejs');
app.use(express.static('public'));

app.get('/', (req, res) => {
    res.render('index');  // renders index.ejs
});

app.listen(port, () => {
    console.log(`Node app running at http://localhost:${port}`);
});
