from src.answer_generator_optimized import AnswerGeneratorOptimized

# 模拟检索结果
mock_context = [
    {
        "text": "Company: Tesla, Inc\nYear: 2023\nRevenue: $96.8 billion\nNet Income: $15.0 billion\nEmployees: 140,000\nMarket Share: 18.5%\nStock Price: $248.48",
        "page": 1
    },
    {
        "text": "R&D Expense: $3.2 billion\nVehicle Deliveries: 1.8 million",
        "page": 2
    },
    {
        "text": "Revenue Growth: 17.7%\nNet Income Growth: 19.2%",
        "page": 3
    }
]

# 测试问题
test_queries = [
    "特斯拉2023年营收是多少",
    "特斯拉2023年净利润是多少",
    "特斯拉2023年员工数量是多少",
    "特斯拉2023年市场份额是多少",
    "特斯拉2023年股价是多少",
    "特斯拉2023年研发支出是多少",
    "特斯拉2023年收入增长率是多少",
    "特斯拉2023年营收增长率是多少",
    "特斯拉2023年净利润增长率是多少",
    "特斯拉2023年利润增长率是多少",
    "特斯拉2023年车辆交付量是多少",
    "特斯拉2023年营收增长了多少",
    "特斯拉2023年净利润增长了多少"
]

# 初始化 AnswerGeneratorOptimized
print("Initializing AnswerGeneratorOptimized...")
generator = AnswerGeneratorOptimized()

# 测试每个查询
print("\nTesting AnswerGeneratorOptimized with various queries...")
for query in test_queries:
    print(f"\nQuery: {query}")
    try:
        result = generator.generate_answer(query=query, context=mock_context, schema="text")
        print(f"Answer: {result.answer}")
        print(f"Reasoning: {result.reasoning}")
        print(f"Citations: {result.citations}")
        print(f"Confidence: {result.confidence}")
        print(f"Validation passed: {result.validation_passed}")
    except Exception as e:
        print(f"Error: {str(e)}")

print("\nTesting completed!")
