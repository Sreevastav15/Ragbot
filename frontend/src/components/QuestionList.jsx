function QuestionList({ questions, onSelect }) {
  if (!questions?.length) return null;

  return (
    <div className="p-4 w-full ">
      <h2 className="text-lg font-semibold mb-3">Suggested Questions</h2>
      <ul className="space-y-2">
        {questions.map((q, i) => (
          <li
            key={i}
            onClick={() => onSelect(q)}
            className="cursor-pointer p-2 border rounded-xl hover:bg-blue-100 transition hover:scale-105"
          >
            {q}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default QuestionList;
