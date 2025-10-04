// frontend/src/components/UploadForm.tsx
import { useRef } from 'react';

type Props = {
  onSubmit: (file: File) => void;
  loading: boolean;
  files: string[];
};

export function UploadForm({ onSubmit, loading, files }: Props) {
  const inputRef = useRef<HTMLInputElement | null>(null);

  const handleChange = (evt: React.ChangeEvent<HTMLInputElement>) => {
    const file = evt.target.files?.[0];
    if (file) onSubmit(file);
  };

  return (
    <div className="upload-card">
      <input
        ref={inputRef}
        type="file"
        accept="application/pdf"
        onChange={handleChange}
        style={{ display: 'none' }}
      />
      <button onClick={() => inputRef.current?.click()} disabled={loading}>
        {loading ? 'Uploading…' : 'Upload PDF'}
      </button>
      {files.length > 0 && (
        <ul className="upload-file-list">
          {files.map((name) => (
            <li key={name}>{name}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
