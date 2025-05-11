action_prompt = (
    "You are a reasoning web GUI agent. In this UI screenshot, I want you to continue executing the command \"{text}\", with the action history being \"{action_history}\".\n"
    "Please provide the action to perform (enumerate in [\"click\", \"select\", \"type\"]), the point where the cursor is moved to (integer), and any value required to complete the action.\n"
    "Output the the final answer in as follows:\n"
    "```json{\"action_type\": enum[\"click\", \"select\", \"type\"], \"click_point\": [x, y], \"value\": \"text to type, or option to select\"}```\n"
    "Note:\n \"click_point\" is needed for all action types, \"value\" is needed for \"type\" and \"select\" action types.\n"
    "Example:\n"
    "```json{\"action_type\": \"click\", \"click_point\": [100, 150], \"value\": \"\"}```\n"
    "```json{\"action_type\": \"type\", \"click_point\": [200, 300], \"value\": \"Who is the president of the United States?\"}```\n"
    "```json{\"action_type\": \"select\", \"click_point\": [150, 200], \"value\": \"Option 1\"}```\n"
)


prompt_critic_system = """As an expert in web interaction and reinforcement learning, you will receive textual descriptions of history interactions for a given web task. You need to evaluate the current action, similar to what a value function does in reinforcement learning. Detailed criteria and standards are given below.

## Explanation of the input content:
1. Task: Brief description of the current web task, such as "Search for a product on an e-commerce website".
2. Description of History operation
   Contains 3 types of web actions. Specific fields and their meanings are as follows:
   [1] CLICK: Click on a web element at a specific position. The "click_point" is represented by a two-dimensional array indicating the absolute position of the click in pixels, such as [100, 150].
      - example: "action_type": "click", "click_point": [100, 150]
   [2] TYPE: Click and input text into a field at a specific position. The "click_point" is represented by a two-dimensional array indicating the absolute position of the click in pixels.
      - example: "action_type": "type", "click_point": [200, 300], "value": "search term"
   [3] SELECT: Click at a specific position to open a dropdown menu, then select an option. Note: The dropdown options may not be visible before clicking, and the "value" field represents the option that will appear and be selected only after the dropdown is opened. The "click_point" is represented by a two-dimensional array indicating the absolute position of the click in pixels.
      - example: "action_type": "select", "click_point": [150, 200], "value": "Qween"
3. A corresponding screenshot of each operation on the current page. The "click_point" position of current action is marked with a semi-transparent red dot in the image.

## Evaluation Criteria:
Here are the detailed descriptions of the two levels. Attention needs to be paid to whether the action taken based on the current screenshot promotes efficient task execution, rather than the relevance of the content shown in the current screenshot to the task:
   Level 1: The action is not the optimal choice for completing the task at this moment, which may lead to deviations from the task flow. For example:
      (1) Clicking the wrong element.
      (2) Typing incorrect or irrelevant text.
      (3) Selecting an incorrect dropdown option.
   Level 2: The action is the optimal and correct choice for completing the task at this moment. For example:
      (1) Clicking the correct button or link to proceed.
      (2) Typing the correct text into the appropriate field.
      (3) Selecting the correct dropdown option.

## Output requirements: 1 or 2 (INT)

## Example Input:
Task Requirements: Search for "laptop" on an e-commerce website.
Previous Action:
step 0: "action_type": "click", "click_point": [120, 40]
step 1: "action_type": "type", "click_point": [300, 400], "value": "laptop"
Current Action and Screenshot:
step 2: "action_type": "click", "click_point": [350, 400]

## Example Output:
2

"""

prompt_critic_user = """Task Requirements: {}
Previous Action: 
{}
Current Action and Screenshot: 
<image>
{}
"""


prompt_policy_system = """You are an expert web automation agent. Given a goal and a sequence of previous actions, your task is to decide the next optimal action to achieve the goal efficiently.

## Action Space:
You can choose from the following action types:
1. CLICK: Click on a web element at a specific position.
   - Fields: "action_type": "click", "click_point": [x, y]
   - "click_point" is a two-dimensional array with absolute pixel values, representing the coordinates on the screen.
2. TYPE: Click and input text into a field at a specific position.
   - Fields: "action_type": "type", "click_point": [x, y], "value": "<text>"
   - "click_point" as above; "value" is the text to input.
3. SELECT: Click at a specific position to open a dropdown menu, then select an option.
   - Fields: "action_type": "select", "click_point": [x, y], "value": "<option>"
   - "click_point" as above; "value" is the dropdown option to select.

## Input:
- Goal: The target to achieve on the web page.
- Previous Actions: The sequence of actions already taken, each in the above format.
- Current Screenshot: The current state of the web page.

## Output requirements:
- Output a single action in JSON format, using only the fields described above. Do not include any explanation or extra text.

## Example Input:
Goal: Search for "laptop" on an e-commerce website.
Previous Actions:
step 0: {"action_type": "click", "click_point": [120, 40]}
step 1: {"action_type": "type", "click_point": [300, 400], "value": "laptop"}

## Example Output:
{"action_type": "click", "click_point": [350, 400]}
"""

prompt_policy_user = """Goal: {}
Previous Actions:
{}
"""



prompt_score_system ="""As an expert in web interaction and reinforcement learning, you will receive a complete sequence of web interaction steps and corresponding descriptions for a given task. You need to evaluate a specific step in terms of its value within the task chain, similar to a value function in reinforcement learning. Detailed criteria and standards are given below.

## Explanation of the input content:
1. Task: Brief description of the current web task, such as "Search for a product on an e-commerce website".
2. Complete operation description and corresponding sequence for the task:
   (1) Text description of operations: Contains 3 types of web actions. Specific fields and their meanings are as follows:
      [1] CLICK: Click on a web element at a specific position. The "click_point" is represented by a two-dimensional array indicating the absolute position of the click in pixels.
         - example: "action_type": "click", "click_point": [100, 150]
      [2] TYPE: Click and input text into a field at a specific position. The "click_point" is represented by a two-dimensional array indicating the absolute position of the click in pixels.
         - example: "action_type": "type", "click_point": [200, 300], "value": "search term"
      [3] SELECT: Click at a specific position to open a dropdown menu, then select an option. Note: The dropdown options may not be visible before clicking, and the "value" field represents the option that will appear and be selected only after the dropdown is opened. The "click_point" is represented by a two-dimensional array indicating the absolute position of the click in pixels.
         - example: "action_type": "select", "click_point": [150, 200], "value": "Qween"
   (2) A corresponding screenshot of each operation on the current page. The "click_point" position of current action is marked with a semi-transparent red dot in the image.
3. The current action to be evaluated and the corresponding screenshot. Please note that you only need to evaluate the current Action (just one step within the complete operation sequence).

## Evaluation Criteria:
Focus on whether the action taken at the current step efficiently promotes task completion, not just its relevance to the current page:
   Level 1: The action is not the optimal choice for completing the task at this moment, which may lead to deviations from the task flow. For example:
      (1) Clicking the wrong element.
      (2) Typing incorrect or irrelevant text.
      (3) Selecting an incorrect dropdown option.
   Level 2: The action is the optimal and correct choice for completing the task at this moment. For example:
      (1) Clicking the correct button or link to proceed.
      (2) Typing the correct text into the appropriate field.
      (3) Selecting the correct dropdown option.

## Output requirements:
- Format: {"rating": int, "explanation": str}. Do not include any additional characters beyond this format.
- The "rating" field should be 1 or 2, indicating the evaluation level. The "explanation" field should explain the reasoning for this rating, without referencing any operations after the current step (future actions are unknown).

## Example Input:
Task Requirements: Search for "laptop" on an e-commerce website.
Action and Screenshot:
step 0: "action_type": "click", "click_point": [120, 40]
step 1: "action_type": "type", "click_point": [300, 400], "value": "laptop"
step 2: "action_type": "click", "click_point": [350, 400]
Current Action(to be evaluated):
step 1: "action_type": "type", "click_point": [300, 400], "value": "laptop"

## Example Output:
{"rating": 2, "explanation": "The action of typing 'laptop' into the search field is the correct and optimal choice for completing the task of searching for a laptop on an e-commerce website. This action directly contributes to achieving the task goal."}

"""

prompt_score_user = """Task Requirements: {}
Action and ScreenShot: {}
Current Action(to be evaluated): 
{}
"""


prompt_negative_system = """As an expert in web interaction and negative sample data constructor, you need to generate a new negative sample of the current action based on historical screenshots and corresponding action descriptions, task description, and the original current action. Detailed criteria and standards are given below.

## Explanation of the input content:
1. Task: Brief description of the current web task, such as "Search for a product on an e-commerce website".
2. History operation description and corresponding screenshot sequence for the task:
   (1) Text description of operations: Contains 3 types of web actions. Specific fields and their meanings are as follows:
      [1] CLICK: Click on a web element at a specific position. The "click_point" is represented by a two-dimensional array indicating the absolute position of the click in pixels.
         - example: "action_type": "click", "click_point": [100, 150]
      [2] TYPE: Click and input text into a field at a specific position. The "click_point" is represented by a two-dimensional array indicating the absolute position of the click in pixels.
         - example: "action_type": "type", "click_point": [200, 300], "value": "search term"
      [3] SELECT: Click at a specific position to open a dropdown menu, then select an option. Note: The dropdown options may not be visible before clicking, and the "value" field represents the option that will appear and be selected only after the dropdown is opened. The "click_point" is represented by a two-dimensional array indicating the absolute position of the click in pixels.
         - example: "action_type": "select", "click_point": [150, 200], "value": "Qween"
   (2) A corresponding screenshot of each operation on the current page. The "click_point" position of current action is marked with a semi-transparent red dot in the image.
3. The positive current action and the corresponding screenshot.

## Criteria for generating negative samples:
The given input is a positive current action that meets the Level 2 standard below. To conduct data augmentation, we need to generate its corresponding negative current action, i.e., the action described below as level 1.
   Level 1: The action is not the optimal choice for completing the task at this moment, which may lead to deviations from the task flow. For example:
      (1) Clicking the wrong element.
      (2) Typing incorrect or irrelevant text.
      (3) Selecting an incorrect dropdown option.
   Level 2: The action is the optimal and correct choice for completing the task at this moment. For example:
      (1) Clicking the correct button or link to proceed.
      (2) Typing the correct text into the appropriate field.
      (3) Selecting the correct dropdown option.

## Output requirements:
- Format: {"action_desc": dict, "explanation": str}. Do not include any additional characters beyond this format.
- The "action_desc" field needs to provide the fields involved in the newly generated negative sample action according to the text description given above. The "explanation" field needs to explain the logic for giving this new negative sample.

## Example Input:
Task Requirements: Search for "laptop" on an e-commerce website.
Previous Action and Screenshot:
step 0: "action_type": "click", "click_point": [120, 40]
step 1: "action_type": "type", "click_point": [300, 400], "value": "laptop"
Origin Action:
step 2: "action_type": "click", "click_point": [350, 400]

## Example Output 1:
{
   "action_desc": {"action_type": "click", "click_point": [900, 100]},
   "explanation": "Instead of clicking the search button to submit the query, clicking a random area on the page will not help complete the search task and may deviate from the task flow."
}

## Example Output 2:
{
   "action_desc": {"action_type": "type", "click_point": [300, 400], "value": "asdfgh"},
   "explanation": "Typing irrelevant text into the search field instead of the correct query will not help achieve the task goal."
}
"""

prompt_negative_user = """Task Requirements: {}
Previous Action and Screenshot: {}
Origin Action: {}
"""

