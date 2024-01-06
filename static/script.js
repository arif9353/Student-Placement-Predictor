const questions= [
    {
        question: "A, B, and C can together do some work in 72 days. A and B can together do two times as much work as C alone, and A and C together can do four times as much work as B alone. Find the time taken by C alone to do the whole work.",
        answers: [
            { text: "144 days", correct: false},
            { text: "360 days", correct: false},
            { text: "216 days", correct: true},
            { text: "180 days", correct: false},
        ]
    },
    {
        question: "A sum of Rs 2387 is divided into three parts in such a way that one-fifth of the first part, one half of the second part and the fourth one and the third part are equal. Find the sum of five times the first part, three times the second part and four times the third part (in rupees).",
        answers: [
            { text: "9982", correct: false},
            { text: "7812", correct: false},
            { text: "9914", correct: false},
            { text: "10199", correct: true},
        ]
    },
    {
        question: "In a test with 26 questions, five points were deducted for each wrong answer and eight points were added for every correct answer. How many were answered correctly if the score was zero?",
        answers: [
            { text: "11", correct: false},
            { text: "10", correct: true},
            { text: "13", correct: false},
            { text: "12", correct: false},
        ]
    },
    {
        question: "A road network covers some cities. City C can be reached only from the city A or city B. The distance from A to C is 65 km and that from B to C is 30 km. The shortest distance from A to B is 58 km. The shortest distance from city P to A is 420 km and the shortest distance from city P to B is 345 km. The shortest distance from city P to city C in kms is:",
        answers: [
            { text: "143", correct: false},
            { text: "478", correct: false},
            { text: "403", correct: false},
            { text: "375", correct: true},
        ]
    },
    {
        question: "5 men and 12 boys finish a piece of work in 4 days, 7 men and 6 boys do it in 5 days. The ratio between the efficiencies of a man and boy is?",
        answers: [
            { text: "1:2", correct: false},
            { text: "2:1", correct: false},
            { text: "2:3", correct: false},
            { text: "6:5", correct: true},
        ]
    },
    {
        question: "The radius of a circle is increased by 1%. Find how much % does its area increases?",
        answers: [
            { text: "1.01%", correct: false},
            { text: "5.01%", correct: false},
            { text: "3.01%", correct: false},
            { text: "2.01%", correct: true},
        ]
    },
    {
        question: "How many number plates can be made if the number plates have two letters of the English alphabets (A-Z) followed by two digits (0-9) if the repetition of digits or alphabets is not allowed?",
        answers: [
            { text: "58500", correct: true},
            { text: "56500", correct: false},
            { text: "52500", correct: false},
            { text: "56800", correct: false},
        ]
    },
    {
        question: "Eesha father was 34 years of age when she was born. Her younger brother, Shashank, now that he is 13, is very proud of the fact that he is as tall as her, even though he is three years younger than her. Eesha’s mother, who is shorter than Eesha, was only 29 when Shashank was born. What is the sum of the ages of Eesha’s parents now?",
        answers: [
            { text: "66", correct: false},
            { text: "76", correct: false},
            { text: "92", correct: true},
            { text: "89", correct: false},
        ]
    },
    {
        question: "The man sitting next to me on the plane was very nervous. He..............................before.",
        answers: [
            { text: "didn't fly", correct: false},
            { text: "hadn't flown", correct: true},
            { text: "hasn't flown", correct: false},
        ]
    },
    {
        question: "Where is the book I gave you? What...............................with it?",
        answers: [
            { text: "have you been doing", correct: false},
            { text: "are you doing", correct: false},
            { text: "have you done", correct: true},
        ]
    },
    {
        question: "Paul lost his watch. Well, if he had looked after it, he ... it.",
        answers: [
            { text: "won't lose", correct: false},
            { text: "wouldn't lose", correct: false},
            { text: "wouldn't have lost", correct: true},
        ]
    },
    {
        question: "By noon, Joe ......... for three hours.",
        answers: [
            { text: "is going to fish", correct: false},
            { text: "will be fishing", correct: false},
            { text: "will have been fishing", correct: true},
        ]
    },
    {
        question: "I haven't seen Mark for weeks. Well, I …. him this afternoon. Why don't you come along?",
        answers: [
            { text: "am meeting", correct: true},
            { text: "have met", correct: false},
            { text: "meet", correct: false},
        ]
    },
    {
        question: "I ….. to reach Jane on the phone all day. Don't you know? She's gone on holiday.",
        answers: [
            { text: "tried", correct: false},
            { text: "have been trying", correct: true},
            { text: "have tried", correct: false},
        ]
    },
    {
        question: "In the following Java code, which code fragment should be inserted at line 3 so that the output will be: '123abc 123abc'?\n1. StringBuilder sb1 = new StringBuilder('123');\n2. String s1 = '123';\n3. // insert code here\n4. System.out.println(sb1 + ' ' + s1);",
        answers: [   
            { text: "sb1.append('abc');s1.append('abc');", correct: false},
            { text: "sb1.append('abc');s1.concat('abc');", correct: false},
            { text: "sb1.concat('abc');s1.append('abc');", correct: false},
            { text: "sb1.append('abc');s1 = s1.concat('abc');", correct: true},
        ]
    },
    {
        question: "What will be the output of the following Java code?\nclass Output\n{\n   public static void main(String args[])\n    {\n         double x = 3.14;\n          int y = (int) Math.abs(x);\n        System.Out.println(y);\n    }\n}",
        answers: [   
            { text: "0", correct: false},
            { text: "3", correct: true},
            { text: "3.0", correct: false},
            { text: "3.1", correct: false},
        ]
    },
    {
        question: "What is the time complexity of the following code?\nPublic boolean isBalanced(String exp)\n{\nint len = exp.length();\nStack<Integer> stk = new Stack<Integer>();\nfor(int i = 0; i < len; i++)\n{\nchar ch = exp.charAt(i);\nif(ch == '(' )\nstk.push(i);\nelse if (ch == ')' )\n{\nif(stk.peek() == null)\n{\nreturn false;\n}\nstk.pop();\n}\n}\n return true;\n}",
        answers: [   
            { text: "O(logn)", correct: false},
            { text: "O(n)", correct: true},
            { text: "O(1)", correct: false},
            { text: "O(nlogn)", correct: false},
        ]
    },
    {
        question: "What is the time complexity for inserting an alphabet in the tree using hash maps?",
        answers: [   
            { text: "O(logn!)", correct: false},
            { text: "O(n!)", correct: false},
            { text: "O(n^2)", correct: false},
            { text: "O(1)", correct: true},
        ]
    }, 
    {
        question: "What will be the output of the following C++ code?\n#include <iostream>\nusing namespace std;\nint operate (int a, int b)\n{\nreturn(a/b);\n}\nint main ()\n{\nint x =5, y = 2;\nfloat n = 5.0, m = 2.0;\ncout << operate(x, y);\ncout << operate (n, m)'\nreturn 0;\n}",
        answers: [   
            { text: "119", correct: false},
            { text: "102.5", correct: true},
            { text: "123.4", correct: false},
            { text: "128.4", correct: false},
        ]
    }, 
    {
        question: "What will be the output of the following C code if the current system date is 6/22/2017?\n#include<stdio.h>\n#include<stdlib.h>#include<time.h>\nint main()\n{\ntime_t ct;\ntime(&ct);\nstruct tm *mt=localtime(&ct);\nprintf('%d',mt-> tm_mon+2);\n}",
        answers: [   
            { text: "8", correct: false},
            { text: "7", correct: true},
            { text: "5", correct: false},
            { text: "6", correct: false},
        ]
    } 
];

const questionElement = document.getElementById("question");
const answerButtons = document.getElementById("answer-buttons");
const nextButton = document.getElementById("next-btn");
const finish = document.getElementById("finish");

let currentQuestionIndex = 0;
let score = 0;

function startQuiz(){
    currentQuestionIndex = 0;
    score = 0;
    nextButton.innerHTML = "Next";
    showQuestion();
}

function showQuestion(){
    resetState();
    let currentQuestion = questions[currentQuestionIndex];
    let questionNo = currentQuestionIndex + 1;
    questionElement.innerHTML = questionNo + ". " + currentQuestion.question.replace(/\n/g, "<br>");

    currentQuestion.answers.forEach(answer => {
        const button = document.createElement("button");
        button.innerHTML = answer.text;
        button.classList.add("btn");
        answerButtons.appendChild(button);
        if(answer.correct){
            button.dataset.correct = answer.correct;
        }
        button.addEventListener("click", selectAnswer);
    });
}

function resetState(){
    nextButton.style.display = "none";
    while(answerButtons.firstChild){
        answerButtons.removeChild(answerButtons.firstChild);    
    }
}

function selectAnswer(e){
    const selectedBtn = e.target;
    const isCorrect = selectedBtn.dataset.correct === "true";
    if(isCorrect){
        selectedBtn.classList.add("correct");
        score++;
    }else{
        selectedBtn.classList.add("incorrect");
    }
    Array.from(answerButtons.children).forEach(button => {
        if(button.dataset.correct === "true"){
            button.classList.add("correct");
        }
        button.disabled = true;
    });
    nextButton.style.display="block";
}





function handleNextButton(){
    currentQuestionIndex++;
    if(currentQuestionIndex < questions.length){
        showQuestion();
    }    else{
        passValue();
        }
}

nextButton.addEventListener("click", ()=>{
    if(currentQuestionIndex < questions.length){
        handleNextButton();
    }
});

function passValue() {
   var variableValue = score.valueOf();
   localStorage.setItem('variableName', variableValue);
//    $.ajax({
//     url: '/form',   
//     type: 'POST',
//     data: { 'data': variableValue },
//     success: function(response) {
    location.href = '/form';
//         console.log(variableValue);
//     },
//     error: function(error) {
//         console.log(error);
//     }
//     });
  }

startQuiz();