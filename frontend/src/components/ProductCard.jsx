function ProductCard({ product }) {
  // Safely parse categories if it's a string
  const getCategories = () => {
    if (!product.categories) return []
    if (Array.isArray(product.categories)) return product.categories
    if (typeof product.categories === 'string') {
      try {
        // Try to parse as JSON array
        const parsed = JSON.parse(product.categories.replace(/'/g, '"'))
        return Array.isArray(parsed) ? parsed : []
      } catch {
        // If parsing fails, split by comma
        return product.categories.split(',').map(c => c.trim()).filter(Boolean)
      }
    }
    return []
  }

  const categories = getCategories()

  return (
    <div className="bg-white border border-gray-200 rounded-lg overflow-hidden hover:shadow-lg transition-shadow duration-300">
      {/* Product Image */}
      <div className="relative h-48 bg-gray-100">
        {product.images && product.images.length > 0 ? (
          <img
            src={product.images[0]}
            alt={product.title || 'Product'}
            className="w-full h-full object-cover"
            onError={(e) => {
              e.target.src = 'https://via.placeholder.com/400x300?text=No+Image'
            }}
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center text-gray-400">
            No Image Available
          </div>
        )}
        {product.similarity_score && (
          <div className="absolute top-2 right-2 bg-indigo-600 text-white px-2 py-1 rounded text-xs font-semibold">
            {(product.similarity_score * 100).toFixed(0)}% Match
          </div>
        )}
      </div>

      {/* Product Details */}
      <div className="p-4">
        <h3 className="text-lg font-semibold text-gray-900 mb-1 line-clamp-2">
          {product.title || 'Untitled Product'}
        </h3>
        
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-gray-500">{product.brand || 'Unknown Brand'}</span>
          {product.price && (
            <span className="text-lg font-bold text-indigo-600">{product.price}</span>
          )}
        </div>

        {categories.length > 0 && (
          <div className="flex flex-wrap gap-1 mb-2">
            {categories.slice(0, 2).map((category, index) => (
              <span
                key={index}
                className="px-2 py-1 bg-gray-100 text-gray-600 text-xs rounded-full"
              >
                {category}
              </span>
            ))}
          </div>
        )}

        {product.generated_description && (
          <div className="mt-3 pt-3 border-t border-gray-200">
            <p className="text-sm text-gray-600 line-clamp-3">
              {product.generated_description}
            </p>
          </div>
        )}

        {product.description && !product.generated_description && (
          <div className="mt-3 pt-3 border-t border-gray-200">
            <p className="text-sm text-gray-600 line-clamp-3">
              {product.description}
            </p>
          </div>
        )}

        <button className="mt-4 w-full bg-indigo-600 text-white py-2 px-4 rounded-lg hover:bg-indigo-700 transition-colors duration-200">
          View Details
        </button>
      </div>
    </div>
  )
}

export default ProductCard
