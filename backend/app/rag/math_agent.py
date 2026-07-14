
def clean_value(value, steps):
    """
    Converts a DocFinQA program value into a number.

    Values like "#0" reference a previous step answer, while values like
    "const_100" are constants from the program.
    """
    if value.startswith("#"):
        return steps[int(value.strip("#"))]["answer"]
    if value.startswith("const_"):
        return float(value.replace("const_", ""))
    return float(value)


def math_agent(question, chunks, dataset="docfinqa", program=None):
    """
    Executes the math program for a supported financial QA example.

    For DocFinQA, this walks through each program step, resolves references
    to earlier calculations, performs the operation, and returns the final answer.
    """

    if dataset != "docfinqa" or program is None:
        return None

    if dataset == "docfinqa":
        steps = program.split("), ")

        listo = []

        for step in steps:
            operation, nums = step.split("(", 1)
            nums = nums.rstrip(")")
            num1, num2 = [num.strip() for num in nums.split(",")]

            listo.append(
                {
                    "operation": operation,
                    "num1": num1,
                    "num2": num2,
                    "answer": 0
                }
            )
        
        for index, data in enumerate(listo):

            #if a number starts with #
            #use that number as the index to get the answer then set that answer to the num1 or 2
            num1 = clean_value(data["num1"], listo)
            num2 = clean_value(data["num2"], listo)


            #cycle through list to perform each operation on the numbers
            if data["operation"] == "add":
                data["answer"] = num1 + num2
            elif data["operation"] == "subtract":
                data["answer"] = num1 - num2
            elif data["operation"] == "multiply":
                data["answer"] = num1 * num2
            elif data["operation"] == "divide":
                if num2 == 0:
                    return None
                data["answer"] = num1 / num2
            elif data["operation"] == "greater":
                data["answer"] = max(num1, num2)
            elif data["operation"] == "exp":
                data["answer"] = num1 ** num2
           
        
    elif dataset == "financebench":
        z=0

    return listo[-1]["answer"]
