function value = W_Rook(n)
    % Make a chessboard (n*n)
    if floor(sqrt(n))==sqrt(n)
        chessboard = zeros(sqrt(n),sqrt(n));
        temp = [1:1:n^2];
        for i=1:sqrt(n)    
            d=1+(i-1)*sqrt(n);
            for j=1:sqrt(n)
                chessboard(i,j)=temp(d);
                d=d+1;
            end
        end
    else 
        error('Error: Square root of an input must be integer.'); 
    end   

    % Contiguity entries by Rook contiguity
    W=zeros(n,n);
    for j=1:sqrt(n) % column-wise
        for i=1:sqrt(n)
            if j==1
                if i==1
                    index=[chessboard(i,j+1) chessboard(i+1,j)];
                    W(chessboard(i,j),index)=1;
                elseif i==sqrt(n)
                    index=[chessboard(i-1,j) chessboard(i,j+1)];
                    W(chessboard(i,j),index)=1;
                else
                    index=[chessboard(i-1,j) chessboard(i,j+1) chessboard(i+1,j)];
                    W(chessboard(i,j),index)=1;
                end
                
            elseif j==sqrt(n)
                if i==1
                    index=[chessboard(i,j-1) chessboard(i+1,j)];
                    W(chessboard(i,j),index)=1;
                elseif i==sqrt(n)
                    index=[chessboard(i,j-1) chessboard(i-1,j)];
                    W(chessboard(i,j),index)=1;
                else
                    index=[chessboard(i,j-1) chessboard(i-1,j) chessboard(i+1,j)];
                    W(chessboard(i,j),index)=1;
                end
     
            else
                if i==1
                   index=[chessboard(i,j-1) chessboard(i,j+1) chessboard(i+1,j)];
                   W(chessboard(i,j),index)=1;
                elseif i==sqrt(n)
                   index=[chessboard(i,j-1) chessboard(i-1,j) chessboard(i,j+1)];
                   W(chessboard(i,j),index)=1;
                else
                   index=[chessboard(i-1,j) chessboard(i,j-1) chessboard(i,j+1) chessboard(i+1,j)];
                   W(chessboard(i,j),index)=1;
                end
            end
        end
    end              
                    
   

value = W;
end