// frontend/src/components/UploadForm.tsx
import { useRef } from 'react';

type Props = { onSubmit: (file: File) => void; loading: boolean };

export function UploadForm({ onSubmit, loading }: Props) {
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
    </div>
  );
}