import json
file_path = 'faq.txt'
with open(file_path, 'r') as file:
    lines = file.readlines()

faq = []


i = 0
while i < len(lines):

    # if not lines[i].strip() or not lines[i].strip().startswith(str(i) + '.'):
    #     i += 1
    #     continue
    

    question = lines[i].split('.', 1)[1].strip().rstrip('?')

    if i + 1 < len(lines):
        answer = lines[i + 1].strip()
    else:
        answer = ''

    faq.append({
        'question': question,
        'answer': answer
    })

    i += 2




output_file = 'faq_output.json'

with open(output_file, 'w') as json_file:
    json.dump(faq, json_file, indent=4, ensure_ascii=False)

print(f"FAQ data has been saved to {output_file}")
