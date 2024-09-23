"use client";
import { useState } from 'react';

// Função para criar uma matriz 10x10 vazia (valores inicializados em 0)
const createEmptyMatrix = () => Array.from({ length: 10 }, () => Array(10).fill(0));

// Componente Matriz interativa
interface MatrixProps {
  matrix: number[][];
  setMatrix: (index: number, newMatrix: number[][]) => void;
  index: number;
}

const Matrix = ({ matrix, setMatrix, index }: MatrixProps) => {
  const toggleCell = (rowIndex: number, colIndex: number) => {
    const newMatrix = matrix.map((row, rIndex) =>
      row.map((cell, cIndex) =>
        rIndex === rowIndex && cIndex === colIndex ? (cell === 1 ? 0 : 1) : cell
      )
    );
    setMatrix(index, newMatrix);
  };

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(10, 20px)', gap: '2px' }}>
      {matrix.map((row, rowIndex) =>
        row.map((cell, colIndex) => (
          <div
            key={`${rowIndex}-${colIndex}`}
            onClick={() => toggleCell(rowIndex, colIndex)}
            style={{
              width: '20px',
              height: '20px',
              backgroundColor: cell === 1 ? 'black' : 'white',
              border: '1px solid black',
              cursor: 'pointer',
            }}
          />
        ))
      )}
    </div>
  );
};

interface TestMatrixProps {
  testMatrix: number[][];
  setTestMatrix: (newMatrix: number[][]) => void;
}

const TestMatrix = ({ testMatrix, setTestMatrix }: TestMatrixProps) => {
  const toggleCell = (rowIndex: number, colIndex: number) => {
    const newMatrix = testMatrix.map((row, rIndex) =>
      row.map((cell, cIndex) =>
        rIndex === rowIndex && cIndex === colIndex ? (cell === 1 ? 0 : 1) : cell
      )
    );
    setTestMatrix(newMatrix);
  };

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(10, 20px)', gap: '2px' }}>
      {testMatrix.map((row, rowIndex) =>
        row.map((cell, colIndex) => (
          <div
            key={`${rowIndex}-${colIndex}`}
            onClick={() => toggleCell(rowIndex, colIndex)}
            style={{
              width: '20px',
              height: '20px',
              backgroundColor: cell === 1 ? 'black' : 'white',
              border: '1px solid black',
              cursor: 'pointer',
            }}
          />
        ))
      )}
    </div>
  );
};

export default function Home() {
  const [matrices, setMatrices] = useState<number[][][]>(Array(5).fill(createEmptyMatrix()));
  const [labels, setLabels] = useState<string[]>(Array(5).fill(''));
  const [testMatrix, setTestMatrix] = useState<number[][]>(createEmptyMatrix());

  const setMatrix = (index: number, newMatrix: number[][]) => {
    const updatedMatrices = matrices.map((matrix, i) => (i === index ? newMatrix : matrix));
    setMatrices(updatedMatrices);
  };

  const handleSubmit = async () => {
    if (labels.some((label) => label === '')) {
      alert('Por favor, preencha todos os campos com as letras correspondentes.');
      return;
    }

    console.log('Matrizes antes do envio:', matrices);

    const response = await fetch('http://localhost:8000/letras/train/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ matrices, labels }),
    });

    const result = await response.json();
    alert(`Treinamento realizado: ${result.message}`);
  };

  const getCookie = (name: string): string | null => {
    let cookieValue: string | null = null;
    if (document.cookie && document.cookie !== '') {
      const cookies = document.cookie.split(';');
      for (let i = 0; i < cookies.length; i++) {
        const cookie = cookies[i].trim();
        if (cookie.substring(0, name.length + 1) === (name + '=')) {
          cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
          break;
        }
      }
    }
    return cookieValue;
  };

  const handlePredict = async () => {
    console.log(testMatrix);

    // Verifique se a matriz está correta
    if (testMatrix.length !== 10 || testMatrix[0].length !== 10) {
        console.error("A matriz deve ter exatamente 10x10 elementos.");
        alert("A matriz deve ter exatamente 10x10 elementos.");
        return;
    }

    console.log(testMatrix)

    const response = await fetch('http://localhost:8000/letras/predict/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ matrix: testMatrix }),
    });

    if (!response.ok) {
        const errorText = await response.text();
        console.error('Erro na resposta do servidor:', errorText);
        alert('Erro na previsão, veja o console para mais detalhes.');
        return;
    }

    const result = await response.json();
    alert(`Letra prevista: ${result.letra_predita}`);
};



  return (
    <div style={{ padding: '20px' }}>
      <h1 style={{ textAlign: 'center' }}>Treinar IA com 5 Matrizes de Letras</h1>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '20px' }}>
        {matrices.map((matrix, index) => (
          <div key={index} style={{ margin: '0 10px', flex: '1' }}>
            <h3>Matriz {index + 1}</h3>
            <Matrix matrix={matrix} setMatrix={setMatrix} index={index} />
            <div style={{ marginTop: '10px' }}>
              <label>
                Letra correspondente:{' '}
                <input
                  type="text"
                  maxLength={1}
                  value={labels[index]}
                  onChange={(e) => {
                    const updatedLabels = labels.map((label, i) =>
                      i === index ? e.target.value.toUpperCase() : label
                    );
                    setLabels(updatedLabels);
                  }}
                />
              </label>
            </div>
          </div>
        ))}
      </div>
      <button onClick={handleSubmit} style={{ marginTop: '20px', padding: '10px 20px' }}>
        Enviar para Treinamento
      </button>

      <div style={{ marginTop: '30px' }}>
        <h3>Matriz de Teste</h3>
        <TestMatrix testMatrix={testMatrix} setTestMatrix={setTestMatrix} />
        <button onClick={handlePredict} style={{ marginTop: '20px', padding: '10px 20px' }}>
          Prever Letra
        </button>
      </div>
    </div>
  );
}


