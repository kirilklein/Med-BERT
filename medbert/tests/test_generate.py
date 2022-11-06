from medbert.data import generate

generator = generate.DataGenerator(100, 2, 5, 1, 10, 1, 30, 10000)
print("[pid, los, icd10, visit_num]")
print([hist for hist in generator.simulate_data()][-1])
