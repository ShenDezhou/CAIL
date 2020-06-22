from pyrouge import Rouge155
r = Rouge155()

r.system_dir = 'system'
r.model_dir = 'model'
r.system_filename_pattern = 'gold_sum.(\d+).txt'
r.model_filename_pattern = 'model_sum.#ID#.txt'

output = r.convert_and_evaluate()
print(output)
output_dict = r.output_to_dict(output)
