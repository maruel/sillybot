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
		input    string
		urgent   bool
		wantt    string
		wantrest string
	}{
		{"", false, "", ""}, // 0
		{"", true, "", ""},
		{"Hi", false, "", "Hi"},
		{"Hi", true, "", "Hi"},
		{"Hi fellow kids!", false, "", "Hi fellow kids!"},
		{"Hi fellow kids!", true, "Hi fellow kids!", ""},
		{"This is code:\n```", false, "", "This is code:\n```"},
		{"This is code:\n```", true, "This is code:\n", "```"},
		{"This is code:\nexit 1", false, "", "This is code:\nexit 1"},
		{"This is code:\nexit 1", true, "This is code:\n", "exit 1"},
		{"This is code:\n```bash\n", false, "", "This is code:\n```bash\n"}, // 10
		{"This is code:\n```bash\n", true, "This is code:\n", "```bash\n"},
		{"This is code:\n```bash\nexit 1\n```", false, "This is code:\n```bash\nexit 1\n```", ""},
		{"This is code:\n```bash\nexit 1\n```", true, "This is code:\n```bash\nexit 1\n```", ""},
		{"This is code:\n```bash\nexit 1\n```\nAnd happiness", false, "This is code:\n```bash\nexit 1\n```", "\nAnd happiness"},
		{"This is code:\n```bash\nexit 1\n```\nAnd happiness", true, "This is code:\n```bash\nexit 1\n```", "\nAnd happiness"},
		{"This is enumeration:\n1. ", false, "", "This is enumeration:\n1. "},
		{"This is enumeration:\n1. ", true, "This is enumeration:\n", "1. "},
		{"1. Do stuff.", false, "", "1. Do stuff."},
		{"1. Do stuff.", true, "1. ", "Do stuff."},
		{"1. Do stuff.\n", false, "", "1. Do stuff.\n"}, // 20
		{"1. Do stuff.\n", true, "1. Do stuff.\n", ""},
		{"1. Do.", false, "", "1. Do."},
		{"1. Do.", true, "", "1. Do."},
		{"- Do stuff.", false, "", "- Do stuff."},
		{"- Do stuff.", true, "- Do stuff.", ""},
		// Do not split on escaped content.
		{"To do what you want, use `os.ReadFile", false, "", "To do what you want, use `os.ReadFile"},
		{"To do what you want, use `os.ReadFile", true, "", "To do what you want, use `os.ReadFile"},
		// Do not split on "node.js"
		{"To do what you want, use node.js and it's going to be fine", false, "", "To do what you want, use node.js and it's going to be fine"},
		{"To do what you want, use node.js and it's going to be fine", true, "", "To do what you want, use node.js and it's going to be fine"},
		{"To do what you want, use Go and it's going to be fine\n\nHello!\nThis is", false, "To do what you want, use Go and it's going to be fine\n", "\nHello!\nThis is"}, // 30
		{"To do what you want, use Go and it's going to be fine\n\nHello!\nThis is", true, "To do what you want, use Go and it's going to be fine\n", "\nHello!\nThis is"},
		{
			" Here is a simple example of a snake game written in Python. This program is a text-based version and does not involve graphics.\n\n```python\nimport random\n\ndef print_screen(snake):\n    for y, row in enumerate(snake):\n",
			false,
			" Here is a simple example of a snake game written in Python. This program is a text-based version and does not involve graphics.\n",
			"\n```python\nimport random\n\ndef print_screen(snake):\n    for y, row in enumerate(snake):\n",
		},
		{
			" Here is a simple example of a snake game written in Python. This program is a text-based version and does not involve graphics.\n\n```python\nimport random\n\ndef print_screen(snake):\n    for y, row in enumerate(snake):\n",
			true,
			" Here is a simple example of a snake game written in Python. This program is a text-based version and does not involve graphics.\n",
			"\n```python\nimport random\n\ndef print_screen(snake):\n    for y, row in enumerate(snake):\n",
		},
		{
			"```python\n" + strings.Repeat("123456789\nabcdefgh\n\n", 110) + "```",
			false,
			"```python\n" + strings.Repeat("123456789\nabcdefgh\n\n", 99) + "```",
			"```python\n" + strings.Repeat("123456789\nabcdefgh\n\n", 11) + "```",
		},
		{
			"```python\n" + strings.Repeat("123456789\nabcdefgh\n\n", 110) + "```",
			true,
			"```python\n" + strings.Repeat("123456789\nabcdefgh\n\n", 99) + "```",
			"```python\n" + strings.Repeat("123456789\nabcdefgh\n\n", 11) + "```",
		},
		{
			" Creating a complete Snake game in Java is beyond the scope of this platform, but I can certainly provide you with a basic outline and some code snippets to help you get started on creating a simple Snake game in Java using the console as the UI.\n\nHere's a high-level structure of the program:\n\n1. Import required libraries\n2. Create a game class to represent the game and its states\n3. Create a game board with methods for rendering and updating the game state\n4. Create a game controller to handle user inputs and manage the game logic\n5. Implement game-specific functions such as moving the snake, eating the apple, and scoring\n\n" +
				"```java\nimport java.util.Random;\nimport java.util.Scanner;\n\npublic class SnakeGame {\n\n    // Game board\n    private char[][] board;\n    private int width;\n    private int height;\n    private char snake;\n    private char apple;\n    private int snakeX, snakeY;\n    private int appleX, appleY;\n    private Direction direction;\n\n    // Game Controller\n    private Scanner scanner;\n    private Random random;\n\n    public SnakeGame(int width, int height) {\n        this.width = width;\n        this.height = height;\n        this.board = new char[height][width];\n        this.scanner = new Scanner(System.in);\n        this.random = new Random();\n\n        initializeGame();\n    }\n\n    private void initializeGame() {\n        // Initialize game board, snake, apple, and direction\n        // ...\n    }\n\n    private void render() {\n        // Render the game board and the current game state\n        // ...\n    }\n\n    private void update() {\n        // Update the game state based on user inputs and game logic\n        // ...\n    }\n\n    private void handleInput() {\n        // Read user input and update direction accordingly\n        // ...\n    }\n\n    private void checkGameOver() {\n        // Check for game over conditions and handle game over logic\n        // ...\n    }\n\n    public void start() {\n        while (!isGameOver()) {\n            handleInput();\n            update();\n            render();\n        }\n    }\n\n    public boolean isGameOver() {\n        // Implement game over conditions here\n        // ...\n    }\n\n    // Other utility functions such as moveSnake, eatApple, etc.\n    // ...\n\n    public static void main(String[] args) {\n        SnakeGame game = new SnakeGame(20, 20);\n        game.start();\n    }\n}\n```",
			false,
			" Creating a complete Snake game in Java is beyond the scope of this platform, but I can certainly provide you with a basic outline and some code snippets to help you get started on creating a simple Snake game in Java using the console as the UI.\n\nHere's a high-level structure of the program:\n\n1. Import required libraries\n2. Create a game class to represent the game and its states\n3. Create a game board with methods for rendering and updating the game state\n4. Create a game controller to handle user inputs and manage the game logic\n5. Implement game-specific functions such as moving the snake, eating the apple, and scoring\n\n" +
				"```java\nimport java.util.Random;\nimport java.util.Scanner;\n\npublic class SnakeGame {\n\n    // Game board\n    private char[][] board;\n    private int width;\n    private int height;\n    private char snake;\n    private char apple;\n    private int snakeX, snakeY;\n    private int appleX, appleY;\n    private Direction direction;\n\n    // Game Controller\n    private Scanner scanner;\n    private Random random;\n\n    public SnakeGame(int width, int height) {\n        this.width = width;\n        this.height = height;\n        this.board = new char[height][width];\n        this.scanner = new Scanner(System.in);\n        this.random = new Random();\n\n        initializeGame();\n    }\n\n    private void initializeGame() {\n        // Initialize game board, snake, apple, and direction\n        // ...\n    }\n\n    private void render() {\n        // Render the game board and the current game state\n        // ...\n    }\n\n    private void update() {\n        // Update the game state based on user inputs and game logic\n        // ...\n    }\n\n    private void handleInput() {\n        // Read user input and update direction accordingly\n        // ...\n    }\n\n    private void checkGameOver() {\n        // Check for game over conditions and handle game over logic\n        // ...\n    }\n\n```",
			"```java\n    public void start() {\n        while (!isGameOver()) {\n            handleInput();\n            update();\n            render();\n        }\n    }\n\n    public boolean isGameOver() {\n        // Implement game over conditions here\n        // ...\n    }\n\n    // Other utility functions such as moveSnake, eatApple, etc.\n    // ...\n\n    public static void main(String[] args) {\n        SnakeGame game = new SnakeGame(20, 20);\n        game.start();\n    }\n}\n```",
		},
		{
			" Creating a complete Snake game in Java is beyond the scope of this platform, but I can certainly provide you with a basic outline and some code snippets to help you get started on creating a simple Snake game in Java using the console as the UI.\n\nHere's a high-level structure of the program:\n\n1. Import required libraries\n2. Create a game class to represent the game and its states\n3. Create a game board with methods for rendering and updating the game state\n4. Create a game controller to handle user inputs and manage the game logic\n5. Implement game-specific functions such as moving the snake, eating the apple, and scoring\n\n" +
				"```java\nimport java.util.Random;\nimport java.util.Scanner;\n\npublic class SnakeGame {\n\n    // Game board\n    private char[][] board;\n    private int width;\n    private int height;\n    private char snake;\n    private char apple;\n    private int snakeX, snakeY;\n    private int appleX, appleY;\n    private Direction direction;\n\n    // Game Controller\n    private Scanner scanner;\n    private Random random;\n\n    public SnakeGame(int width, int height) {\n        this.width = width;\n        this.height = height;\n        this.board = new char[height][width];\n        this.scanner = new Scanner(System.in);\n        this.random = new Random();\n\n        initializeGame();\n    }\n\n    private void initializeGame() {\n        // Initialize game board, snake, apple, and direction\n        // ...\n    }\n\n    private void render() {\n        // Render the game board and the current game state\n        // ...\n    }\n\n    private void update() {\n        // Update the game state based on user inputs and game logic\n        // ...\n    }\n\n    private void handleInput() {\n        // Read user input and update direction accordingly\n        // ...\n    }\n\n    private void checkGameOver() {\n        // Check for game over conditions and handle game over logic\n        // ...\n    }\n\n    public void start() {\n        while (!isGameOver()) {\n            handleInput();\n            update();\n            render();\n        }\n    }\n\n    public boolean isGameOver() {\n        // Implement game over conditions here\n        // ...\n    }\n\n    // Other utility functions such as moveSnake, eatApple, etc.\n    // ...\n\n    public static void main(String[] args) {\n        SnakeGame game = new SnakeGame(20, 20);\n        game.start();\n    }\n}\n```",
			true,
			" Creating a complete Snake game in Java is beyond the scope of this platform, but I can certainly provide you with a basic outline and some code snippets to help you get started on creating a simple Snake game in Java using the console as the UI.\n\nHere's a high-level structure of the program:\n\n1. Import required libraries\n2. Create a game class to represent the game and its states\n3. Create a game board with methods for rendering and updating the game state\n4. Create a game controller to handle user inputs and manage the game logic\n5. Implement game-specific functions such as moving the snake, eating the apple, and scoring\n\n" +
				"```java\nimport java.util.Random;\nimport java.util.Scanner;\n\npublic class SnakeGame {\n\n    // Game board\n    private char[][] board;\n    private int width;\n    private int height;\n    private char snake;\n    private char apple;\n    private int snakeX, snakeY;\n    private int appleX, appleY;\n    private Direction direction;\n\n    // Game Controller\n    private Scanner scanner;\n    private Random random;\n\n    public SnakeGame(int width, int height) {\n        this.width = width;\n        this.height = height;\n        this.board = new char[height][width];\n        this.scanner = new Scanner(System.in);\n        this.random = new Random();\n\n        initializeGame();\n    }\n\n    private void initializeGame() {\n        // Initialize game board, snake, apple, and direction\n        // ...\n    }\n\n    private void render() {\n        // Render the game board and the current game state\n        // ...\n    }\n\n    private void update() {\n        // Update the game state based on user inputs and game logic\n        // ...\n    }\n\n    private void handleInput() {\n        // Read user input and update direction accordingly\n        // ...\n    }\n\n    private void checkGameOver() {\n        // Check for game over conditions and handle game over logic\n        // ...\n    }\n\n```",
			"```java\n    public void start() {\n        while (!isGameOver()) {\n            handleInput();\n            update();\n            render();\n        }\n    }\n\n    public boolean isGameOver() {\n        // Implement game over conditions here\n        // ...\n    }\n\n    // Other utility functions such as moveSnake, eatApple, etc.\n    // ...\n\n    public static void main(String[] args) {\n        SnakeGame game = new SnakeGame(20, 20);\n        game.start();\n    }\n}\n```",
		},
	}
	for i, line := range data {
		t.Run(strconv.Itoa(i), func(t *testing.T) {
			if gott, gotrest := splitResponse(line.input, line.urgent); line.wantt != gott || line.wantrest != gotrest {
				t.Fatalf("%q %t\nWant: %q\nGot:  %q\nWant: %q\nGot:  %q", line.input, line.urgent, line.wantt, gott, line.wantrest, gotrest)
			}
		})
	}
}
