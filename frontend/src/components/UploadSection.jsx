import { FileText } from "lucide-react";

function UploadSection({ fileName }) {
  // Don't render anything until upload is complete
  if (!fileName) return null;

  return (
    <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-4 w-[320px] transition-transform duration-200 hover:scale-[1.02] z-50">
      <div className="flex items-start gap-2 animate-fadeIn">
        <FileText className="text-blue-600 w-5 h-5 flex-shrink-0 mt-[2px]" />
        <div className="flex flex-col overflow-hidden">
          <span className="font-semibold text-gray-900 text-sm">Uploaded:</span>
          <span
            className="font-medium text-gray-700 text-sm break-words overflow-hidden text-ellipsis whitespace-normal"
          >
            {fileName}
          </span>
        </div>
      </div>
    </div>
  );
}

export default UploadSection;
