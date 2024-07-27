// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package main

import (
	"strconv"
	"strings"
	"testing"
)

func TestSplitResponse(t *testing.T) {
	data := []struct {
		input  string
		urgent bool
		want   int
	}{
		{"", false, 0},
		{"", true, 0},
		{"Hi", false, 0},
		{"Hi", true, 0},
		{"Hi!", false, 0},
		{"Hi!", true, 0},
		{"Hi fellow kids!", false, 0},
		{"Hi fellow kids!", true, 15},
		{"This is code:\n```", false, 0},
		{"This is code:\n```", true, 14},
		{"This is code:\nFoo", false, 0},
		{"This is code:\nFoo", true, 14},
		{"This is code:\n```Foo", false, 0},
		{"This is code:\n```Foo", true, 14},
		{"This is code:\n``Foo```", false, 0},
		{"This is code:\n``Foo```", true, 14},
		{"This is code:\n```Foo```", true, 23},
		{"This is code:\n```Foo```", false, 23},
		{"This is code:\n```Foo```\nAnd happiness", false, 23},
		{"This is code:\n```Foo```\nAnd happiness", true, 23},
		{"This is enumeration:\n1. ", false, 0},
		{"This is enumeration:\n1. ", true, 21},
		{"1. Do stuff.", false, 0},
		{"1. Do stuff.", true, 3},
		{"1. Do stuff.\n", false, 0},
		{"1. Do stuff.\n", true, 13},
		{"1. Do.", false, 0},
		{"1. Do.", true, 0},
		{"- Do stuff.", false, 0},
		{"- Do stuff.", true, 11},
		{"To do what you want, use `os.ReadFile", false, 0},
		{"To do what you want, use `os.ReadFile", true, 0},
		// Do not split on" node.js"
		{"To do what you want, use node.js and it's going to be fine", false, 0},
		{"To do what you want, use node.js and it's going to be fine", true, 0},
		{"To do what you want, use Go and it's going to be fine\n\nHello!\nThis is", false, 54},
		{"To do what you want, use Go and it's going to be fine\n\nHello!\nThis is", true, 54},
		{
			" Here is a simple example of a snake game written in Python. This program is a text-based version and does not involve graphics.\n\n```python\nimport random\n\ndef print_screen(snake):\n    for y, row in enumerate(snake):\n",
			true,
			129,
		},
		{
			" Here is a simple example of a snake game written in Python. This program is a text-based version and does not involve graphics.\n\n```python\nimport random\n\ndef print_screen(snake):\n    for y, row in enumerate(snake):\n",
			false,
			129,
		},
		{
			"```" + strings.Repeat("123456789\n12345678\n\n", 110) + "```",
			false,
			1982,
		},
		{
			"```" + strings.Repeat("123456789\n12345678\n\n", 110) + "```",
			true,
			1982,
		},
		{
			" Creating a complete Snake game in Java is beyond the scope of this platform, but I can certainly provide you with a basic outline and some code snippets to help you get started on creating a simple Snake game in Java using the console as the UI.\n\nHere's a high-level structure of the program:\n\n1. Import required libraries\n2. Create a game class to represent the game and its states\n3. Create a game board with methods for rendering and updating the game state\n4. Create a game controller to handle user inputs and manage the game logic\n5. Implement game-specific functions such as moving the snake, eating the apple, and scoring\n\n```java\nimport java.util.Random;\nimport java.util.Scanner;\n\npublic class SnakeGame {\n\n    // Game board\n    private char[][] board;\n    private int width;\n    private int height;\n    private char snake;\n    private char apple;\n    private int snakeX, snakeY;\n    private int appleX, appleY;\n    private Direction direction;\n\n    // Game Controller\n    private Scanner scanner;\n    private Random random;\n\n    public SnakeGame(int width, int height) {\n        this.width = width;\n        this.height = height;\n        this.board = new char[height][width];\n        this.scanner = new Scanner(System.in);\n        this.random = new Random();\n\n        initializeGame();\n    }\n\n    private void initializeGame() {\n        // Initialize game board, snake, apple, and direction\n        // ...\n    }\n\n    private void render() {\n        // Render the game board and the current game state\n        // ...\n    }\n\n    private void update() {\n        // Update the game state based on user inputs and game logic\n        // ...\n    }\n\n    private void handleInput() {\n        // Read user input and update direction accordingly\n        // ...\n    }\n\n    private void checkGameOver() {\n        // Check for game over conditions and handle game over logic\n        // ...\n    }\n\n    public void start() {\n        while (!isGameOver()) {\n            handleInput();\n            update();\n            render();\n        }\n    }\n\n    public boolean isGameOver() {\n        // Implement game over conditions here\n        // ...\n    }\n\n    // Other utility functions such as moveSnake, eatApple, etc.\n    // ...\n\n    public static void main(String[] args) {\n        SnakeGame game = new SnakeGame(20, 20);\n        game.start();\n    }\n}\n```",
			false,
			1893,
		},
		{
			" Creating a complete Snake game in Java is beyond the scope of this platform, but I can certainly provide you with a basic outline and some code snippets to help you get started on creating a simple Snake game in Java using the console as the UI.\n\nHere's a high-level structure of the program:\n\n1. Import required libraries\n2. Create a game class to represent the game and its states\n3. Create a game board with methods for rendering and updating the game state\n4. Create a game controller to handle user inputs and manage the game logic\n5. Implement game-specific functions such as moving the snake, eating the apple, and scoring\n\n```java\nimport java.util.Random;\nimport java.util.Scanner;\n\npublic class SnakeGame {\n\n    // Game board\n    private char[][] board;\n    private int width;\n    private int height;\n    private char snake;\n    private char apple;\n    private int snakeX, snakeY;\n    private int appleX, appleY;\n    private Direction direction;\n\n    // Game Controller\n    private Scanner scanner;\n    private Random random;\n\n    public SnakeGame(int width, int height) {\n        this.width = width;\n        this.height = height;\n        this.board = new char[height][width];\n        this.scanner = new Scanner(System.in);\n        this.random = new Random();\n\n        initializeGame();\n    }\n\n    private void initializeGame() {\n        // Initialize game board, snake, apple, and direction\n        // ...\n    }\n\n    private void render() {\n        // Render the game board and the current game state\n        // ...\n    }\n\n    private void update() {\n        // Update the game state based on user inputs and game logic\n        // ...\n    }\n\n    private void handleInput() {\n        // Read user input and update direction accordingly\n        // ...\n    }\n\n    private void checkGameOver() {\n        // Check for game over conditions and handle game over logic\n        // ...\n    }\n\n    public void start() {\n        while (!isGameOver()) {\n            handleInput();\n            update();\n            render();\n        }\n    }\n\n    public boolean isGameOver() {\n        // Implement game over conditions here\n        // ...\n    }\n\n    // Other utility functions such as moveSnake, eatApple, etc.\n    // ...\n\n    public static void main(String[] args) {\n        SnakeGame game = new SnakeGame(20, 20);\n        game.start();\n    }\n}\n```",
			true,
			1893,
		},
		{
			" Creating a complete Snake game in Java is quite extensive and beyond the scope of this platform. However, I can provide you with a simple implementation of the Snake game using JavaFX, which will give you a good starting point.\n\nFirstly, you will need to make sure that you have JavaFX installed on your system. If you don't have it, you can download it from the official website: https://gluonhq.com/products/javafx/\n\nNow, create a new JavaFX project and follow these steps to create a simple Snake game:\n\n1. Import necessary packages:\n\n```java\nimport javafx.application.Application;\nimport javafx.scene.Scene;\nimport javafx.scene.control.Label;\nimport javafx.scene.layout.BorderPane;\nimport javafx.scene.layout.GridPane;\nimport javafx.scene.paint.Color;\nimport javafx.scene.shape.Rectangle;\nimport javafx.stage.Stage;\nimport java.util.ArrayList;\nimport java.util.Random;\n\npublic class SnakeGame extends Application {\n    // ...\n}\n```\n\n2. Define necessary variables for the game:\n\n```java\nprivate static final int WIDTH = 640;\nprivate static final int HEIGHT = 640;\nprivate int[][] gameBoard;\nprivate Snake snake;\nprivate Food food;\nprivate boolean running = false;\n```\n\n3. Initialize the game board, snake, and food:\n\n```java\n@Override\npublic void start(Stage stage) {\n    // ...\n\n    gameBoard = new int[30][WIDTH / 20];\n    snake = new Snake(3);\n    food = new Food();\n    // ...\n}\n```\n\n4. Create `Snake` and `Food` classes to handle snake movement and positioning:\n\n```java\nclass Snake {\n    private int[] bodyParts;\n    // ...\n}\n\nclass Food {\n    private int x;\n    private int y;\n    // ...\n}\n```\n\n5. Add methods for key event handling:\n\n```java\npublic void keyPressed(KeyEvent event) {\n    // Handle direction changes based on user input\n}\n```\n\n6. Create methods for updating the game:\n\n```java\npublic void update() {\n    // Update game state\n}\n```\n\n7. Create methods for rendering the game on the screen:\n\n```java\npublic void draw() {\n    // Clear the screen\n    // Draw the game elements (snake, food, score, game over message, etc.)\n}\n```\n\n8. Use the `update` and `draw` methods to animate the game in the game loop:\n\n```java\n@Override\npublic void run() {\n    while (running) {\n        update();\n        draw();\n        try {\n            Thread.sleep(100);\n        } catch (InterruptedException e) {\n            e.printStackTrace();\n        }\n    }\n}\n```\n\n9. Start the game loop and display the window:\n\n```java\nnew Thread(this::run).start();\n\nScene scene = new Scene(gridPane, WIDTH, HEIGHT);\nstage.setTitle(\"Snake Game\");\nstage.setScene(scene);\nstage.show();\n```\n\nThis code provides you with a simple Snake game implementation in JavaFX. To make the game more engaging, consider adding features like speed increases, power-ups, and game over screen with a score. Don't forget to import the required JavaFX packages and include a main method to launch the application.",
			false,
			936,
		},
	}
	for i, line := range data {
		t.Run(strconv.Itoa(i), func(t *testing.T) {
			if got := splitResponse(line.input, line.urgent); line.want != got {
				t.Fatalf("%q: %d != %d\nWant: %q\nGot:  %q", line.input, line.want, got, line.input[:line.want], line.input[:got])
			}
		})
	}
}
